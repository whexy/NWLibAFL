//! The RLFuzzer schedulers are a family of schedulers that use the bandit
//! algorithm to get a better result of corpus selecting.

use alloc::borrow::ToOwned;
use core::marker::PhantomData;

use super::testcase_score::{
    GeneratedTestcaseScore, SeedSelectionTestcaseScore, SeedTimeTestcaseScore,
};
use crate::{
    corpus::Corpus,
    inputs::UsesInput,
    schedulers::{Scheduler, TestcaseScore},
    state::{HasCorpus, HasMetadata, UsesState},
    Error,
};

/// Use the bandit algorithm to get a better result of corpus selecting.
#[derive(Debug, Clone)]
pub struct RLScheduler<S, F> {
    phantom: PhantomData<(S, F)>,
}

impl<S, F> UsesState for RLScheduler<S, F>
where
    S: UsesInput,
{
    type State = S;
}

impl<S, F> Scheduler for RLScheduler<S, F>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseScore<Self::State>,
{
    /// Gets the next entry. Every time this function is called, scores of all testcases will be updated.
    fn next(&self, state: &mut Self::State) -> Result<usize, Error> {
        let corpus_num = state.corpus().count();
        if corpus_num == 0 {
            Err(Error::empty("No entries in corpus".to_owned()))
        } else {
            let mut min_score = f64::MAX;
            let mut id = match state.corpus().current() {
                Some(cur) => {
                    if *cur + 1 >= state.corpus().count() {
                        0
                    } else {
                        *cur + 1
                    }
                }
                None => 0,
            };
            // find the seed with smallest score.
            for idx in 0..corpus_num {
                let mut entry = state.corpus().get(idx).unwrap().borrow_mut();
                if let Ok(score) = F::compute(&mut *entry, state) {
                    println!("💯 {}, {}", idx, score);
                    if score < min_score {
                        min_score = score;
                        id = idx;
                    }
                }
            }
            *state.corpus_mut().current_mut() = Some(id);
            println!("🔥 {} selected", id);
            Ok(id)
        }
    }
}

impl<S, F> RLScheduler<S, F>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseScore<S>,
{
    /// Creates a new [`RLScheduler`]
    #[must_use]
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

/// AFLFast-like scheduler, minimize s(i)
pub type SelectionRLScheduler<S> = RLScheduler<S, SeedSelectionTestcaseScore<S>>;

/// BlackboxFuzz-like scheduler, minimize N(i)
pub type GeneratedRLScheduler<S> = RLScheduler<S, GeneratedTestcaseScore<S>>;

/// BlackboxFuzz-like scheduler, minimize t(i)
pub type TotalTimeRLScheduler<S> = RLScheduler<S, SeedTimeTestcaseScore<S>>;
