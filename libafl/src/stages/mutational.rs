//| The [`MutationalStage`] is the default stage used during fuzzing.
//! For the current input, it will perform a range of random mutations, and then run them in the executor.

use alloc::string::ToString;
use core::{marker::PhantomData, slice::from_raw_parts};
use std::hash::Hasher;

use ahash::AHasher;

#[cfg(feature = "introspection")]
use crate::monitors::PerfFeature;
use crate::{
    bolts::{current_time, rands::Rand},
    corpus::{testcase::RLFuzzTestcaseMetaData, Corpus},
    feedbacks::MapIndexesMetadata,
    fuzzer::Evaluator,
    mark_feature_time,
    mutators::Mutator,
    prelude::{rlsched::BanditMetadata, HasMetadata},
    stages::Stage,
    start_timer,
    state::{HasClientPerfMonitor, HasCorpus, HasRand, UsesState},
    Error,
};

// TODO multi mutators stage

/// A Mutational stage is the stage in a fuzzing run that mutates inputs.
/// Mutational stages will usually have a range of mutations that are
/// being applied to the input one by one, between executions.
pub trait MutationalStage<E, EM, M, Z>: Stage<E, EM, Z>
where
    E: UsesState<State = Self::State>,
    M: Mutator<Self::State>,
    EM: UsesState<State = Self::State>,
    Z: Evaluator<E, EM, State = Self::State>,
    Self::State: HasClientPerfMonitor + HasCorpus + HasMetadata,
{
    /// The mutator registered for this stage
    fn mutator(&self) -> &M;

    /// The mutator registered for this stage (mutable)
    fn mutator_mut(&mut self) -> &mut M;

    /// Gets the number of iterations this mutator should run for.
    fn iterations(&self, state: &mut Z::State, corpus_idx: usize) -> Result<u64, Error>;

    /// Runs this (mutational) stage for the given testcase
    #[allow(clippy::cast_possible_wrap)] // more than i32 stages on 32 bit system - highly unlikely...
    fn perform_mutational(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut Z::State,
        manager: &mut EM,
        corpus_idx: usize,
    ) -> Result<(), Error> {
        let num = self.iterations(state, corpus_idx)?;

        for i in 0..num {
            start_timer!(state);
            let mut input = state
                .corpus()
                .get(corpus_idx)?
                .borrow_mut()
                .load_input()?
                .clone();
            mark_feature_time!(state, PerfFeature::GetInputFromCorpus);

            start_timer!(state);
            self.mutator_mut().mutate(state, &mut input, i as i32)?;
            mark_feature_time!(state, PerfFeature::Mutate);

            // Time is measured directly the `evaluate_input` function
            let start = current_time();
            let (result, new_corpus_idx) =
                fuzzer.evaluate_input(state, executor, manager, input)?;
            let once_time = current_time() - start;

            // [current testcase] update corpus index i,
            //                           edge map hash,
            //                           selected times s(i),
            //                           and total time t(i).
            {
                let mut current_testcase = state.corpus().get(corpus_idx)?.borrow_mut();
                if !current_testcase.has_metadata::<RLFuzzTestcaseMetaData>() {
                    let mut rlmeta = RLFuzzTestcaseMetaData::new();
                    *rlmeta.corpus_idx_mut() = corpus_idx;
                    *rlmeta.selected_times_mut() += 1;
                    *rlmeta.total_time_mut() += once_time;

                    let mapmeta = current_testcase
                        .metadata()
                        .get::<MapIndexesMetadata>()
                        .ok_or_else(|| {
                            Error::key_not_found("Failed to get MapIndexesMetadata".to_string())
                        })?;
                    let mut hasher = AHasher::default();
                    let slice = mapmeta.list.as_slice();
                    let ptr = slice.as_ptr() as *const u8;
                    let map_size = slice.len() * std::mem::size_of::<usize>();
                    unsafe {
                        let raw_mem = from_raw_parts(ptr, map_size);
                        hasher.write(raw_mem);
                    }
                    let hash = hasher.finish();
                    *rlmeta.hash_mut() = hash;
                    current_testcase.add_metadata::<RLFuzzTestcaseMetaData>(rlmeta);
                } else {
                    let rlmeta = current_testcase
                        .metadata_mut()
                        .get_mut::<RLFuzzTestcaseMetaData>()
                        .unwrap();
                    *rlmeta.selected_times_mut() += 1;
                    *rlmeta.total_time_mut() += once_time;
                }
            }

            match result {
                crate::ExecuteInputResult::None => {}
                crate::ExecuteInputResult::Corpus | crate::ExecuteInputResult::Solution => {
                    // [current testcase, new path found] update generated N(i).
                    {
                        let mut testcase = state.corpus().get(corpus_idx)?.borrow_mut();
                        let rlmeta = testcase
                            .metadata_mut()
                            .get_mut::<RLFuzzTestcaseMetaData>()
                            .ok_or_else(|| {
                                Error::key_not_found("Failed to get RLFuzzMetadata".to_string())
                            })?;

                        let generated = rlmeta.generated_mut();
                        *generated += 1;

                        println!("😁 Find new path!");
                        println!("\tupdate testcase {}!", corpus_idx);
                        println!("\t{:?}", rlmeta);
                    }

                    {
                        let mut bdmeta = state.metadata_mut().get_mut::<BanditMetadata>().unwrap();
                        bdmeta.reward = true;
                    }
                }
            };

            start_timer!(state);
            self.mutator_mut()
                .post_exec(state, i as i32, new_corpus_idx)?;
            mark_feature_time!(state, PerfFeature::MutatePostExec);
        }
        Ok(())
    }
}

/// Default value, how many iterations each stage gets, as an upper bound.
/// It may randomly continue earlier.
pub static DEFAULT_MUTATIONAL_MAX_ITERATIONS: u64 = 128;

/// The default mutational stage
#[derive(Clone, Debug)]
pub struct StdMutationalStage<E, EM, M, Z> {
    mutator: M,
    #[allow(clippy::type_complexity)]
    phantom: PhantomData<(E, EM, Z)>,
}

impl<E, EM, M, Z> MutationalStage<E, EM, M, Z> for StdMutationalStage<E, EM, M, Z>
where
    E: UsesState<State = Z::State>,
    EM: UsesState<State = Z::State>,
    M: Mutator<Z::State>,
    Z: Evaluator<E, EM>,
    Z::State: HasClientPerfMonitor + HasCorpus + HasRand + HasMetadata,
{
    /// The mutator, added to this stage
    #[inline]
    fn mutator(&self) -> &M {
        &self.mutator
    }

    /// The list of mutators, added to this stage (as mutable ref)
    #[inline]
    fn mutator_mut(&mut self) -> &mut M {
        &mut self.mutator
    }

    /// Gets the number of iterations as a random number
    fn iterations(&self, state: &mut Z::State, _corpus_idx: usize) -> Result<u64, Error> {
        Ok(1 + state.rand_mut().below(DEFAULT_MUTATIONAL_MAX_ITERATIONS))
    }
}

impl<E, EM, M, Z> UsesState for StdMutationalStage<E, EM, M, Z>
where
    E: UsesState<State = Z::State>,
    EM: UsesState<State = Z::State>,
    M: Mutator<Z::State>,
    Z: Evaluator<E, EM>,
    Z::State: HasClientPerfMonitor + HasCorpus + HasRand + HasMetadata,
{
    type State = Z::State;
}

impl<E, EM, M, Z> Stage<E, EM, Z> for StdMutationalStage<E, EM, M, Z>
where
    E: UsesState<State = Z::State>,
    EM: UsesState<State = Z::State>,
    M: Mutator<Z::State>,
    Z: Evaluator<E, EM>,
    Z::State: HasClientPerfMonitor + HasCorpus + HasRand + HasMetadata,
{
    #[inline]
    #[allow(clippy::let_and_return)]
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut Z::State,
        manager: &mut EM,
        corpus_idx: usize,
    ) -> Result<(), Error> {
        let ret = self.perform_mutational(fuzzer, executor, state, manager, corpus_idx);

        #[cfg(feature = "introspection")]
        state.introspection_monitor_mut().finish_stage();

        ret
    }
}

impl<E, EM, M, Z> StdMutationalStage<E, EM, M, Z>
where
    E: UsesState<State = Z::State>,
    EM: UsesState<State = Z::State>,
    M: Mutator<Z::State>,
    Z: Evaluator<E, EM>,
    Z::State: HasClientPerfMonitor + HasCorpus + HasRand,
{
    /// Creates a new default mutational stage
    pub fn new(mutator: M) -> Self {
        Self {
            mutator,
            phantom: PhantomData,
        }
    }
}

#[cfg(feature = "python")]
#[allow(missing_docs)]
/// `StdMutationalStage` Python bindings
pub mod pybind {
    use pyo3::prelude::*;

    use crate::{
        events::pybind::PythonEventManager,
        executors::pybind::PythonExecutor,
        fuzzer::pybind::PythonStdFuzzer,
        mutators::pybind::PythonMutator,
        stages::{pybind::PythonStage, StdMutationalStage},
    };

    #[pyclass(unsendable, name = "StdMutationalStage")]
    #[derive(Debug)]
    /// Python class for StdMutationalStage
    pub struct PythonStdMutationalStage {
        /// Rust wrapped StdMutationalStage object
        pub inner:
            StdMutationalStage<PythonExecutor, PythonEventManager, PythonMutator, PythonStdFuzzer>,
    }

    #[pymethods]
    impl PythonStdMutationalStage {
        #[new]
        fn new(mutator: PythonMutator) -> Self {
            Self {
                inner: StdMutationalStage::new(mutator),
            }
        }

        fn as_stage(slf: Py<Self>) -> PythonStage {
            PythonStage::new_std_mutational(slf)
        }
    }

    /// Register the classes to the python module
    pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PythonStdMutationalStage>()?;
        Ok(())
    }
}
