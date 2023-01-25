//! The RLFuzzer schedulers are a family of schedulers that use the bandit
//! algorithm to get a better result of corpus selecting.

use alloc::vec::Vec;
use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::{
    corpus::{testcase::RLFuzzTestcaseMetaData, Corpus},
    inputs::UsesInput,
    schedulers::Scheduler,
    state::{HasCorpus, HasMetadata, UsesState},
    Error,
};

/// Compute probability distributions over a corpus.
pub trait TestcaseDistribution<S>
where
    S: HasCorpus + HasMetadata,
{
    /// Compute the score of all testcase.
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error>;
}

/// Combine two [`TestcaseDistribution`]s together.
#[derive(Debug, Clone)]
pub struct TestcaseDistributionCombiner<S, F, G>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseDistribution<S>,
    G: TestcaseDistribution<S>,
{
    phantom: PhantomData<(S, F, G)>,
    strategy_num: usize,
}

impl<S, F, G> TestcaseDistributionCombiner<S, F, G>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseDistribution<S>,
    G: TestcaseDistribution<S>,
{
    /// Create a new [`TestcaseDistributionCombiner`].
    pub fn new(strategy_num: usize) -> Self {
        Self {
            phantom: PhantomData,
            strategy_num: strategy_num,
        }
    }

    /// get the number of strategies in the combiner.
    pub fn strategy_num(&self) -> usize {
        self.strategy_num
    }

    /// invoke compute() function of self.
    pub fn get_advice(&self, state: &S) -> Result<Vec<Vec<f64>>, Error> {
        Self::compute(state)
    }
}

impl<S, F, G> TestcaseDistribution<S> for TestcaseDistributionCombiner<S, F, G>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseDistribution<S>,
    G: TestcaseDistribution<S>,
{
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error> {
        let mut metrics = F::compute(state)?;
        metrics.extend(G::compute(state)?);
        Ok(metrics)
    }
}

/// Number of times a seed being selected.
#[derive(Debug, Clone)]
pub struct SeedSelectionDistribution<S> {
    phantom: PhantomData<S>,
}

impl<S> TestcaseDistribution<S> for SeedSelectionDistribution<S>
where
    S: HasCorpus + HasMetadata,
{
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error> {
        let mut metrics = Vec::with_capacity(state.corpus().count());
        for idx in 0..state.corpus().count() {
            let entry = state.corpus().get(idx)?.borrow();
            if entry.has_metadata::<RLFuzzTestcaseMetaData>() {
                let rlmeta = entry.metadata().get::<RLFuzzTestcaseMetaData>().unwrap();
                metrics.push(rlmeta.selected_times() as f64 + 1.0);
            } else {
                metrics.push(1 as f64);
            }
        }
        let sum = metrics.iter().map(|x| 1.0 / x).sum::<f64>();
        Ok(vec![metrics
            .into_iter()
            .map(|x| ((1.0 / x) / sum))
            .collect()])
    }
}

/// Number of inputs generated by fuzzing seed
#[derive(Debug, Clone)]
pub struct SeedGeneratedDistribution<S> {
    phantom: PhantomData<S>,
}

impl<S> TestcaseDistribution<S> for SeedGeneratedDistribution<S>
where
    S: HasCorpus + HasMetadata,
{
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error> {
        let mut metrics = Vec::with_capacity(state.corpus().count());
        for idx in 0..state.corpus().count() {
            let entry = state.corpus().get(idx)?.borrow();
            if entry.has_metadata::<RLFuzzTestcaseMetaData>() {
                let rlmeta = entry.metadata().get::<RLFuzzTestcaseMetaData>().unwrap();
                metrics.push(rlmeta.generated() as f64 + 1.0);
            } else {
                metrics.push(1 as f64);
            }
        }
        let sum = metrics.iter().map(|x| 1.0 / x).sum::<f64>();
        Ok(vec![metrics
            .into_iter()
            .map(|x| ((1.0 / x) / sum))
            .collect()])
    }
}

/// SLIME: M(i)/s(i)
#[derive(Debug, Clone)]
pub struct SlimeDistribution<S> {
    phantom: PhantomData<S>,
}

impl<S> TestcaseDistribution<S> for SlimeDistribution<S>
where
    S: HasCorpus + HasMetadata,
{
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error> {
        let mut metrics = Vec::with_capacity(state.corpus().count());
        for idx in 0..state.corpus().count() {
            let entry = state.corpus().get(idx)?.borrow();
            if entry.has_metadata::<RLFuzzTestcaseMetaData>() {
                let rlmeta = entry.metadata().get::<RLFuzzTestcaseMetaData>().unwrap();
                // add one to make sure we don't generate NaN.
                let generated = rlmeta.generated() + 1;
                let selected_times = rlmeta.selected_times() + 1;
                metrics.push(generated as f64 / selected_times as f64);
            } else {
                metrics.push(1 as f64);
            }
        }
        let sum = metrics.iter().sum::<f64>();
        Ok(vec![metrics.into_iter().map(|x| x / sum).collect()])
    }
}

/// Time spent on each seed
#[derive(Debug, Clone)]
pub struct TimeDistribution<S> {
    phantom: PhantomData<S>,
}

impl<S> TestcaseDistribution<S> for TimeDistribution<S>
where
    S: HasCorpus + HasMetadata,
{
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error> {
        let mut metrics = Vec::with_capacity(state.corpus().count());
        for idx in 0..state.corpus().count() {
            let entry = state.corpus().get(idx)?.borrow();
            if entry.has_metadata::<RLFuzzTestcaseMetaData>() {
                let rlmeta = entry.metadata().get::<RLFuzzTestcaseMetaData>().unwrap();
                metrics.push(rlmeta.total_time().as_secs_f32() as f64 + 1.0);
            } else {
                metrics.push(1 as f64);
            }
        }
        let sum = metrics.iter().map(|x| 1.0 / x).sum::<f64>();
        Ok(vec![metrics
            .into_iter()
            .map(|x| ((1.0 / x) / sum))
            .collect()])
    }
}

/// Expected time to generate new path, M(i)/t(i)
#[derive(Debug, Clone)]
pub struct ExpectedTimeDistribution<S> {
    phantom: PhantomData<S>,
}

impl<S> TestcaseDistribution<S> for ExpectedTimeDistribution<S>
where
    S: HasCorpus + HasMetadata,
{
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error> {
        let mut metrics = Vec::with_capacity(state.corpus().count());
        for idx in 0..state.corpus().count() {
            let entry = state.corpus().get(idx)?.borrow();
            if entry.has_metadata::<RLFuzzTestcaseMetaData>() {
                let rlmeta = entry.metadata().get::<RLFuzzTestcaseMetaData>().unwrap();
                let generated = rlmeta.generated() + 1;
                let total_time = rlmeta.total_time().as_secs_f32() as f64 + 1.0;
                metrics.push(generated as f64 / total_time);
            } else {
                metrics.push(1 as f64);
            }
        }
        let sum = metrics.iter().sum::<f64>();
        Ok(vec![metrics.into_iter().map(|x| (x / sum)).collect()])
    }
}

/// EcoFuzz: fii / sqrt(i)
#[derive(Debug, Clone)]
pub struct EcoFuzzDistribution<S> {
    phantom: PhantomData<S>,
}

impl<S> TestcaseDistribution<S> for EcoFuzzDistribution<S>
where
    S: HasCorpus + HasMetadata,
{
    fn compute(state: &S) -> Result<Vec<Vec<f64>>, Error> {
        let mut metrics = Vec::with_capacity(state.corpus().count());
        for idx in 0..state.corpus().count() {
            let entry = state.corpus().get(idx)?.borrow();
            if entry.has_metadata::<RLFuzzTestcaseMetaData>() {
                let rlmeta = entry.metadata().get::<RLFuzzTestcaseMetaData>().unwrap();
                let generated = rlmeta.generated() + 1;
                let selection = rlmeta.selected_times() + 1;
                let self_transition_ratio = (selection - generated) as f64 / selection as f64;
                metrics.push(1.0 - self_transition_ratio / (idx as f64 + 1.0).sqrt());
            } else {
                metrics.push(1 as f64);
            }
        }
        let sum = metrics.iter().sum::<f64>();
        Ok(vec![metrics.into_iter().map(|x| (x / sum)).collect()])
    }
}

/// Temporary type used for testing, combine all distributions together.
//  TODO: implement a macro for combining distributions.
pub type CombinedDistribution<S> = TestcaseDistributionCombiner<
    S,
    TestcaseDistributionCombiner<
        S,
        TestcaseDistributionCombiner<
            S,
            TestcaseDistributionCombiner<
                S,
                TestcaseDistributionCombiner<
                    S,
                    SeedSelectionDistribution<S>,
                    SeedGeneratedDistribution<S>,
                >,
                SlimeDistribution<S>,
            >,
            TimeDistribution<S>,
        >,
        ExpectedTimeDistribution<S>,
    >,
    EcoFuzzDistribution<S>,
>;

/// The metadata used in bendit algorithm
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BanditMetadata {
    /// weight for different arms
    weights: Vec<f64>,
    /// probability of last round
    prob: Vec<f64>,
    /// advice of last round
    last_advice: Vec<Vec<f64>>,
    /// index of the last selected seed
    last_selected_seed: usize,
    /// if the last seed generated new path
    pub reward: bool,
}

crate::impl_serdeany!(BanditMetadata);

/// Use the bandit algorithm to get a better result of corpus selecting.
#[derive(Debug, Clone)]
pub struct BanditScheduler<S, F, G>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseDistribution<S>,
    G: TestcaseDistribution<S>,
{
    phantom: PhantomData<(S, F, G)>,
    combiner: TestcaseDistributionCombiner<S, F, G>,
    gamma: f64,
}

impl<S, F, G> UsesState for BanditScheduler<S, F, G>
where
    S: UsesInput + HasCorpus + HasMetadata,
    F: TestcaseDistribution<S>,
    G: TestcaseDistribution<S>,
{
    type State = S;
}

impl<S, F, G> Scheduler for BanditScheduler<S, F, G>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseDistribution<S>,
    G: TestcaseDistribution<S>,
{
    /// Gets the next entry. Every time this function is called, scores of all testcases will be updated.
    fn next(&self, state: &mut Self::State) -> Result<usize, Error> {
        let corpus_num = state.corpus().count();
        let advice = self.combiner.get_advice(state)?;

        // init the bandit algorithm
        if !state.has_metadata::<BanditMetadata>() {
            let weights = vec![1.0; self.combiner.strategy_num()];
            state.add_metadata(BanditMetadata {
                weights,
                prob: vec![0.0; corpus_num],
                last_advice: vec![],
                last_selected_seed: 0,
                reward: false,
            })
        } else {
            // update weights with last round information
            let bdmeta = state.metadata_mut().get_mut::<BanditMetadata>().unwrap();
            let weights = &mut bdmeta.weights;
            let last_advice = &bdmeta.last_advice;
            let prob = &bdmeta.prob;
            let reward = if bdmeta.reward { 1.0 } else { 0.0 };
            let reward_hat = reward / prob[bdmeta.last_selected_seed];
            for exp in 0..self.combiner.strategy_num() {
                let y = reward_hat * last_advice[exp][bdmeta.last_selected_seed];
                weights[exp] *= (self.gamma * y / corpus_num as f64).exp();
            }

            if bdmeta.reward {
                for exp in 0..self.combiner.strategy_num() {
                    println!("expert {}, weight={}", exp, weights[exp]);
                }
            }
        }

        let bdmeta = state.metadata_mut().get_mut::<BanditMetadata>().unwrap();

        let prob = &mut bdmeta.prob;
        let weights = &bdmeta.weights;
        // extend prob to the same length of advice
        prob.extend(vec![0.0; advice[0].len() - prob.len()]);
        for idx in 0..corpus_num {
            let mut sum = 0.0;
            let sum_of_weight = weights.iter().sum::<f64>();
            for exp in 0..self.combiner.strategy_num() {
                sum += weights[exp] * advice[exp][idx];
            }
            prob[idx] = (1.0 - self.gamma) * sum / sum_of_weight + self.gamma / corpus_num as f64;
        }

        // print advice tensor
        // println!("😁 BanditScheduler: advice tensor is:");
        // for exp in 0..self.combiner.strategy_num() {
        //     println!("expert {}, weight={}", exp, weights[exp]);
        //     for idx in 0..corpus_num {
        //         println!("\tseed {} is {}", idx, advice[exp][idx]);
        //     }
        // }

        for idx in 0..corpus_num {
            println!("😁 BanditScheduler: prob of seed {} is {}", idx, prob[idx]);
        }

        // select the index of max number from prob
        let mut max = 0.0;
        let mut max_idx = 0;
        for idx in 0..corpus_num {
            if prob[idx] > max {
                max = prob[idx];
                max_idx = idx;
            }
        }

        bdmeta.last_selected_seed = max_idx;
        bdmeta.reward = false;
        bdmeta.last_advice = advice;
        // println!("😁 BanditScheduler: select seed {}", max_idx);
        Ok(max_idx)
    }
}

impl<S, F, G> BanditScheduler<S, F, G>
where
    S: HasCorpus + HasMetadata,
    F: TestcaseDistribution<S>,
    G: TestcaseDistribution<S>,
{
    /// Creates a new [`BanditScheduler`]
    #[must_use]
    pub fn new(combiner: TestcaseDistributionCombiner<S, F, G>) -> Self {
        Self {
            phantom: PhantomData,
            combiner: combiner,
            gamma: 0.1,
        }
    }
}
