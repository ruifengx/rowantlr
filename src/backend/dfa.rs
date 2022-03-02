/*
 * rowantlr: ANTLR-like parser generator framework targeting rowan
 * Copyright (C) 2022  Xie Ruifeng
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


//! DFA related data structures, backend for regular expressions.
//!
//! - Use [`Expr::build`], [`BuildResult::try_resolve`] to build DFAs;
//! - Run a DFA on some input with [`Dfa::run`] and [`Resolved::run`];
//! - Minimise a DFA with [`Resolved::minimise`];
//!
//! Below is an example:
//!
//! ```
//! use rowantlr::ir::lexical::Expr;
//! use rowantlr::backend::dfa::InvalidInput;
//! let expr = Expr::concat([
//!     Expr::singleton('a'),
//!     Expr::many(Expr::union([
//!         Expr::singleton('a'),
//!         Expr::singleton('b'),
//!     ])),
//!     Expr::singleton('b'),
//! ]);
//! let resolved = expr.build().try_resolve().unwrap();
//! // the minimised DFA and the original should behave the same
//! for m in [resolved.clone(), resolved.minimise()] {
//!     assert_eq!(Some(()), m.run("aab".chars()).unwrap());
//!     assert_eq!(Some(()), m.run("ab".chars()).unwrap());
//!     assert_eq!(Some(()), m.run("abab".chars()).unwrap());
//!     assert_eq!(None, m.run("aa".chars()).unwrap());
//!     let InvalidInput { current_state, current_input, remaining_input }
//!         = m.run("baaa".chars()).unwrap_err();
//!     assert_eq!(current_state, 0);
//!     assert_eq!(current_input, 'b');
//!     assert_eq!(remaining_input.as_str(), "aaa");
//! }
//! ```
//!
//! Use [`Expr::build_many`] to build [`Dfa`] from multiple regular expressions, and use
//! [`BuildResult::apply_hint`] to resolve a conflict ([`BuildResult::apply_hints`] can
//! also be used to resolve conflicts in batch):
//!
//! ```
//! use rowantlr::ir::lexical::{Expr, PosInfo};
//! use rowantlr::backend::dfa::{ResolveError, InvalidInput};
//! // start with 'a', end with 'b'
//! let e1 = Expr::concat([
//!     Expr::singleton('a'),
//!     Expr::many(Expr::any_of("ab")),
//!     Expr::singleton('b'),
//! ]);
//! // even number of 'a's and 'b's.
//! let e2 = Expr::many(Expr::union([
//!     Expr::from("aa"),
//!     Expr::from("bb"),
//!     Expr::concat([
//!         Expr::from("ab") | Expr::from("ba"),
//!         Expr::many(Expr::from("aa") | Expr::from("bb")),
//!         Expr::from("ab") | Expr::from("ba"),
//!     ]),
//! ]));
//! // 'e1' and 'e2' should intersect.
//! let result = Expr::build_many([(&e1, 0), (&e2, 1)]);
//! let (result, conflicts) = result.try_resolve().expect_err("conflict expected");
//! // exactly one conflict is detected.
//! assert_eq!(conflicts.len(), 1);
//! let conflict = &conflicts[0];
//! // tag '0' and tag '1' are involved in this conflict.
//! assert_eq!(vec![0, 1], conflict.conflicting_tags(&result).copied().collect::<Vec<_>>());
//! // as for the different interpretations for some problematic input ...
//! conflict.interpretations.iter().for_each(|ps| {
//!     // all interpretations is indeed of the same input "aabb".
//!     assert_eq!("aabb".to_string(), ps.split_last().unwrap().1.iter()
//!         .map(|&p| result.position_info[p as usize].info.into_normal().unwrap())
//!         .collect::<String>());
//!     // each position list is chained according to the 'follow_pos' relation.
//!     ps.windows(2).for_each(|p|
//!         assert!(result.position_info[p[0] as usize].follow_pos.contains(&p[1])));
//! });
//! // apply a hint to resolve the conflict.
//! let mut result = result;
//! match result.apply_hint("aabb".chars(), 0) {
//!     ResolveError::Resolved(s) => {
//!         // we indeed resolved the conflict between tag '0' and tag '1' for input "aabb".
//!         assert_eq!(2, s.len());
//!         assert!(matches!(result.position_info[s[0] as usize].info, PosInfo::Accept(0)));
//!         assert!(matches!(result.position_info[s[1] as usize].info, PosInfo::Accept(1)));
//!     }
//!     err => panic!(r#""aabb" should resolve a conflict, but we got {:?} instead"#, err),
//! }
//! // now we are ready to build the DFA.
//! let resolved = result.try_resolve().expect("no more conflict expected");
//! // the minimised DFA and the original should behave the same
//! for m in [resolved.clone(), resolved.minimise()] {
//!     // the following two inputs have no conflict from the very beginning:
//!     assert_eq!(Some(0), m.run("abb".chars()).unwrap());
//!     assert_eq!(Some(1), m.run("bbaa".chars()).unwrap());
//!     // the hint we used to resolve the conflict behaves well:
//!     assert_eq!(Some(0), m.run("aabb".chars()).unwrap());
//!     // other conflicts are solved just as for the hint:
//!     assert_eq!(Some(0), m.run("abab".chars()).unwrap());
//! }
//! ```
//!
//! The above two example DFAs are actually minimal from the very beginning. Below is an example of
//! how the DFA can have equivalent states and be properly minimised:
//! ```
//! use rowantlr::ir::lexical::Expr;
//! use rowantlr::backend::dfa::InvalidInput;
//! let expr = Expr::union([
//!     Expr::some(Expr::singleton('a')), // a+
//!     Expr::some(Expr::from("aa")),     // (aa)+
//! ]);
//! let resolved = expr.build().try_resolve().unwrap();
//! let minimised = resolved.clone().minimise();
//! // the minimised DFA and the original should behave the same
//! for m in [&resolved, &minimised] {
//!     assert_eq!(Some(()), m.run("a".chars()).unwrap());
//!     assert_eq!(Some(()), m.run("aa".chars()).unwrap());
//!     assert_eq!(Some(()), m.run("aaa".chars()).unwrap());
//!     assert_eq!(None, m.run("".chars()).unwrap());
//! }
//! assert!(minimised.dfa.state_count < resolved.dfa.state_count);
//! assert_eq!(resolved.dfa.state_count, 3);
//! assert_eq!(minimised.dfa.state_count, 2);
//! ```

use std::rc::Rc;
use std::borrow::Borrow;
use std::collections::{BTreeSet, BTreeMap, VecDeque};

use itertools::Itertools;
use derivative::Derivative;

use crate::utils::Dict;
use crate::utils::interval::Intervals;
use crate::utils::partition_refinement::{IndexManager, Partitions};
use crate::ir::lexical::char_class::Char;
use crate::ir::lexical::{Expr, ExprInfo, ExprVisitor, PosInfo};

/// Definite Finite State Automata.
#[derive(Debug, Clone)]
pub struct Dfa<A> {
    /// Number of states in this DFA.
    pub state_count: u32,
    /// Transitions (arcs) of this DFA.
    pub transitions: Dict<(u32, A, u32)>,
}

/// Invalid input for some [`Dfa`].
///
/// Since our transition function of a [`Dfa`] is partial, it is possible that running on some
/// prefix of an input string is undefined.
#[derive(Debug, Eq, PartialEq)]
pub struct InvalidInput<A, I> {
    /// The last state this [`Dfa`] reaches.
    pub current_state: u32,
    /// The input character for which the transition function is not defined.
    pub current_input: A,
    /// Possible remaining input.
    pub remaining_input: I,
}

impl<A: Ord> Dfa<A> {
    /// Run this DFA on the specific `input`, starting from some `start_state`.
    /// If no transition is available, return a tuple of state, input char, and rest input.
    pub fn run<'a, C, I>(&self, start_state: u32, input: I)
                         -> Result<u32, InvalidInput<C, I::IntoIter>>
        where A: 'a, C: 'a + Borrow<A>, I: IntoIterator<Item=C> {
        let mut remaining_input = input.into_iter();
        let mut current_state = start_state;
        for current_input in &mut remaining_input {
            current_state = match self.transitions
                .get((&current_state, current_input.borrow())) {
                Some(p) => *p,
                None => return Err(InvalidInput {
                    current_state,
                    current_input,
                    remaining_input,
                }),
            };
        }
        Ok(current_state)
    }
}

/// Calculate `followpos(p)` for every position `p` in a regular expression `e`.
/// This piece of information can be used to construct a DFA.
#[derive(Derivative)]
#[derivative(Default(bound = ""))]
pub struct Builder<A, Tag>(Vec<PosEntry<A, Tag>>);

/// Information related to a position in a regular expression.
#[derive(Debug)]
pub struct PosEntry<A, Tag> {
    /// Whether this state is an accept state.
    pub info: PosInfo<A, Tag>,
    /// The `followpos(p)` set for this position `p`.
    pub follow_pos: BTreeSet<u32>,
}

impl<A, Tag> PosEntry<A, Tag> {
    /// Returns `true` if `info` is a `PosInfo::Accept`.
    pub fn is_accept(&self) -> bool {
        matches!(&self.info, PosInfo::Accept(_))
    }
}

impl<A, Tag> Builder<A, Tag> {
    /// Create a new [`Builder`], the same as [`Default::default`].
    pub fn new() -> Self { Builder::default() }
    /// Finish by getting the collected information. See also [`Builder::build`].
    pub fn finish(self) -> Vec<PosEntry<A, Tag>> { self.0 }
}

impl<A, Tag> ExprVisitor<A, Tag> for Builder<A, Tag> {
    fn visit_union(&mut self, _: &ExprInfo, _: &ExprInfo) {}
    fn visit_cat(&mut self, lhs: &ExprInfo, rhs: &ExprInfo) {
        for i in lhs.last_pos.iter().copied() {
            self.0[idx!(i)].follow_pos.extend(rhs.first_pos.iter().copied())
        }
    }
    fn visit_some(&mut self, x: &ExprInfo) {
        self.visit_cat(x, x)
    }
    fn gen_info(&mut self, p_info: PosInfo<A, Tag>) -> ExprInfo {
        let info = ExprInfo::singleton(narrow!(self.0.len()));
        self.0.push(PosEntry { info: p_info, follow_pos: BTreeSet::new() });
        info
    }
}

impl<A: Ord + Clone, Tag> Builder<A, Tag> {
    /// Build a [`Dfa`].
    ///
    /// To preserve as much information as possible, the map from DFA states and positions in
    /// regular expressions is provided as is, without substitution with tags in [`PosEntry`]s.
    /// In this form, conflicts can be easily detected, and error reports can be associated with
    /// the original regular expressions.
    pub fn build(self, expr_info: ExprInfo) -> BuildResult<A, Tag> {
        let start = Rc::new(expr_info.first_pos);
        // state -> state index
        let mut state_mapping = BTreeMap::new();
        state_mapping.insert(start.clone(), 0);
        // state index * input -> state index
        let mut state_transition = BTreeMap::new();
        // [state * state index]
        let mut queue = VecDeque::new();
        queue.push_back((start, 0));

        while let Some((q0, k0)) = queue.pop_front() { // q0: state
            debug_assert_eq!(state_mapping.get(&q0), Some(&k0));
            let mut trans = BTreeMap::<A, BTreeSet<u32>>::new();
            for p in q0.iter().copied() { // p: position
                if let PosInfo::Normal(a) = &self.0[idx!(p)].info {
                    trans.entry(a.clone())
                        .or_default()
                        .extend(&self.0[idx!(p)].follow_pos);
                }
            }
            for (a, q) in trans {
                let q = Rc::new(q);
                let k = state_mapping.len();
                let k = *state_mapping.entry(q.clone()).or_insert_with(|| {
                    queue.push_back((q.clone(), k));
                    k
                });
                state_transition.insert((narrow!(k0), a), narrow!(k));
            }
        }

        let state_mapping = {
            let mut result = vec![BTreeSet::new(); state_mapping.len()];
            for (q, k) in state_mapping {
                result[k] = Rc::try_unwrap(q).unwrap();
            }
            result
        };

        BuildResult {
            dfa: Dfa {
                state_count: narrow!(state_mapping.len()),
                transitions: state_transition.into_iter()
                    .map(|((s, a), t)| (s, a, t)).collect(),
            },
            state_mapping,
            position_info: self.0,
        }
    }
}

/// Information for reachability for states in a DFA for a specific [`BuildResult`].
pub struct Reachability<A> {
    /// The mapping is `state ->? (state, position)`:
    /// - `state` for the predecessor state;
    /// - `position` for the position in the original regular expression.
    ///
    /// If `predecessor[s] == Some((t, a))`, we have in turn:
    /// - there is an edge (labelled `a`) from `t` to `s`;
    /// and for each position `q` in `s`, we have such a position `p` that
    /// - `p` is in `state_mapping[t]`;
    /// - `position_info[p] = Normal(a)`;
    /// - `q` is in `followpos(p)`;
    /// The existence of `p` follows immediately from the existence of the edge.
    ///
    /// At most one predecessor is recorded here. If the [`BuildResult`] is obtained by
    /// [`built`](Builder::build) from an [`Expr`](Expr), and if this reachability
    /// information is generated from [`BuildResult::reachability`], follow this predecessor
    /// should give a shortest path from the initial state to the current state.
    pub predecessor: Dict<(u32, (u32, A))>,
}

impl<A: Eq> Reachability<A> {
    /// Try to generate a path from the initial state to a specific state.
    /// Every element in the path is a `(state, input)` pair.
    pub fn try_get_paths_for<Tag>(&self, state: u32, build_result: &BuildResult<A, Tag>)
                                  -> Option<Vec<Vec<u32>>> {
        let mut current_state = state;
        let mut result_paths = build_result.state_mapping[idx!(state)].iter()
            .filter(|&&p| build_result.position_info[idx!(p)].is_accept())
            .map(|&p| vec![p]).collect_vec();
        while let Some((pred, a)) = self.predecessor.get((&current_state, )) {
            for path in &mut result_paths {
                let last_pos = *path.last().unwrap();
                let pos = build_result.state_mapping[idx!(*pred)].iter().copied()
                    .find(|&p| build_result.position_info[idx!(p)].follow_pos.contains(&last_pos)
                        && build_result.position_info[idx!(p)].info.as_ref().into_normal() == Some(a))
                    .unwrap();
                path.push(pos);
            }
            current_state = *pred;
        }
        if current_state != 0 { return None; }
        result_paths.iter_mut().for_each(|path| path.reverse());
        Some(result_paths)
    }
}

/// Result from [`build`](Builder::build)ing a [`Dfa`].
#[derive(Debug)]
pub struct BuildResult<A, Tag> {
    /// The resulted DFA.
    pub dfa: Dfa<A>,
    /// Mapping from DFA states to positions in the original regular expression.
    pub state_mapping: Vec<BTreeSet<u32>>,
    /// Mapping from [`PosInfo::Accept`] positions to their tags.
    pub position_info: Vec<PosEntry<A, Tag>>,
}

impl<A: Ord + Clone, Tag> BuildResult<A, Tag> {
    /// Calculate reachability information for DFA states.
    pub fn reachability(&self) -> Reachability<A> {
        let mut predecessor = vec![None; idx!(self.dfa.state_count)];
        for (s, a, t) in self.dfa.transitions.iter() {
            if s == t || predecessor[idx!(*t)].is_some() { continue; }
            predecessor[idx!(*t)] = Some((*s, a.clone()));
        }
        let predecessor = predecessor.into_iter().zip(0..)
            .filter_map(|(p, k)| Some((k, p?))).collect();
        Reachability { predecessor }
    }
}

/// [`BuildResult`] with conflicts resolved.
#[derive(Debug, Clone)]
pub struct Resolved<A, Tag> {
    /// The resulted DFA.
    pub dfa: Dfa<A>,
    /// Mapping from states to tags.
    pub tags: Dict<(u32, Tag)>,
}

impl<A: Ord, Tag: Clone> Resolved<A, Tag> {
    /// Run the resolved DFA on this specific input and get the result.
    pub fn run<'a, C, I>(&self, input: I) -> Result<Option<Tag>, InvalidInput<C, I::IntoIter>>
        where A: 'a, C: 'a + Borrow<A>, I: IntoIterator<Item=C> {
        let s = self.dfa.run(0, input)?;
        Ok(self.tags.get((&s, )).map(Tag::clone))
    }
}

impl<A: Clone + Ord, Tag: Clone + Ord> Resolved<A, Tag> {
    /// Minimise the DFA using [Hopcroft's algorithm](https://en.wikipedia.org/wiki/DFA_minimization).
    pub fn minimise(self) -> Self {
        let accepts = self.tags.clone().inverse::<1>();
        let reverse_trans = self.dfa.transitions.iter()
            .map(|(s, a, t)| (*t, a.clone(), *s))
            .collect::<Dict<_>>();
        let mut partitions = Partitions::new_trivial(self.dfa.state_count);
        let mut pending = Intervals::new();
        for (_, g) in accepts.groups::<1>() {
            partitions.refine_with(g, &mut pending);
        }
        while let Some(p) = pending.pop_part(&partitions) {
            for (_, g) in partitions[p].iter()
                .flat_map(|t| reverse_trans.equal_range((t, )))
                .map(|(a, s)| (a.clone(), *s))
                .collect::<Dict<_>>()
                .groups::<1>() {
                partitions.refine_with(g, &mut pending);
            }
        }
        partitions.promote_to_head(partitions.parent_part_of(&0));
        let mut transitions = self.dfa.transitions.into_raw();
        for (s, _, t) in transitions.iter_mut() {
            *s = partitions.parent_part_of(s);
            *t = partitions.parent_part_of(t);
        }
        let transitions = Dict::from(transitions);
        let dfa = Dfa { state_count: narrow!(partitions.parts().len()), transitions };
        let mut tags = self.tags.into_raw();
        for (s, _) in tags.iter_mut() {
            *s = partitions.parent_part_of(s);
        }
        let tags = Dict::from(tags);
        Resolved { dfa, tags }
    }
}

impl<A> Dfa<A> {
    /// Group the input tokens by the transition arcs they belong to.
    /// Input `a` on arc `(s, a, t)` is classified according to key `(s, t)`.
    pub fn classify_input<P>(&self, inputs: &mut Partitions<A, P>)
        where A: Ord + Clone, P: IndexManager<A> {
        self.transitions.iter().cloned()
            .map(|(s, a, t)| (s, t, a))
            .collect::<Dict<_>>()
            .groups::<2>()
            .for_each(|(_, g)| inputs.refine_with(g, &mut ()));
    }
}

impl Dfa<u32> {
    /// Refine the inputs of this DFA with the partitions obtained by [`Dfa::classify_input`].
    ///
    /// ```
    /// # use rowantlr::ir::lexical::{Expr, char_class::{Char, CharClass}};
    /// # use rowantlr::utils::partition_refinement::Partitions;
    /// // auxiliary function for constructing a singleton of a char class:
    /// fn char_class<C: Into<CharClass>>(c: C) -> Expr<CharClass> { Expr::singleton(c.into()) }
    /// // a lexical grammar, whose input can be further refined after DFA generation:
    /// let many_a_or_many_b = Expr::union([ // equivalent to '(a | b)*'
    ///     Expr::many(char_class('a')), // a*
    ///     Expr::many(char_class('b')), // b*
    /// ]);
    /// let (expr, mut classifier) = Expr::freeze_char_class([&many_a_or_many_b]);
    /// // 'a' and 'b' are distinguishable at the very beginning
    /// assert_ne!(classifier.classify((&Char::from('a'), )),
    ///            classifier.classify((&Char::from('b'), )));
    /// // resolve, minimise, and refine
    /// let expr = {
    ///     assert_eq!(expr.len(), 1);
    ///     expr.into_iter().next().unwrap()
    /// };
    /// let resolved = expr.build().try_resolve().unwrap();
    /// let minimised = resolved.clone().minimise();
    /// let (refined, refined_classifier) = {
    ///     let (mut minimised, mut classifier) = (minimised.clone(), classifier.clone());
    ///     let input_count = 1 + classifier.values::<1>().copied().max().unwrap();
    ///     let mut inputs = Partitions::new_trivial(input_count);
    ///     minimised.dfa.classify_input(&mut inputs);
    ///     minimised.dfa.refine_input(&inputs, &mut classifier);
    ///     (minimised, classifier)
    /// };
    /// // 'a' and 'b' now becomes indistinguishable according to the refined DFA
    /// assert_eq!(refined_classifier.classify((&Char::from('a'), )),
    ///            refined_classifier.classify((&Char::from('b'), )));
    /// // equivalence of the DFA before and after minimisation, as well as after refinement
    /// for (m, c) in [(&resolved, &classifier),
    ///                (&minimised, &classifier),
    ///                (&refined, &refined_classifier)] {
    ///     let idx = |x: char| c.classify((&Char::from(x), ));
    ///     let run = |s: &str| m.run(s.chars().map(idx)).unwrap();
    ///     assert_eq!(Some(()), run(""));
    ///     assert_eq!(Some(()), run("a"));
    ///     assert_eq!(Some(()), run("aaa"));
    ///     assert_eq!(Some(()), run("bb"));
    /// }
    /// ```
    pub fn refine_input(&mut self, inputs: &Partitions<u32>, classifier: &mut Dict<(Char, u32)>) {
        let mut transitions = std::mem::take(&mut self.transitions).into_raw().into_vec();
        for (_, a, _) in &mut transitions {
            *a = inputs.parent_part_of(a);
        }
        self.transitions = Dict::from(transitions);
        let mut class = std::mem::take(classifier).into_raw().into_vec();
        for (_, k) in &mut class {
            *k = inputs.parent_part_of(k);
        }
        *classifier = Dict::from(class);
    }
}

/// Conflict in DFA states.
#[derive(Debug)]
pub struct Conflict {
    /// An example input string for this conflict, different interpretations provided.
    /// Characters of this input string is encoded as positions on the original regular
    /// expression (instead of the input character `A`) for better error reporting.
    ///
    /// The last position in each position list is guaranteed to be an [`PosInfo::Accept`]ing
    /// position, and can be used to extract the conflicting tags.
    pub interpretations: Box<[Box<[u32]>]>,
}

impl Conflict {
    /// Get the conflicting tags for this conflict.
    pub fn conflicting_tags<'a, A, Tag>(&'a self, build_result: &'a BuildResult<A, Tag>)
                                        -> impl Iterator<Item=&'a Tag> {
        self.interpretations.iter().map(|ps| {
            let p = idx!(*ps.last().unwrap());
            build_result.position_info[p].info.as_ref().into_accept().unwrap()
        })
    }
}

/// Result for resolving conflicts.
#[derive(Debug)]
pub enum ResolveError<A, Tag, Input> {
    /// No error. Original mapping returned.
    Resolved(Box<[u32]>),
    /// There is only one tag for the input string. Thus there is no conflict at all.
    NoConflict(u32),
    /// The state for the input string is not an accepted state. This input was refused.
    NotAcceptState(NotAcceptState<Tag>),
    /// The specified input is not defined for this DFA.
    InvalidInput(InvalidInput<A, Input>),
}

/// Decide what to do next for a non-accepting state.
#[derive(Debug)]
pub struct NotAcceptState<Tag> {
    /// The index of the state in the [`Dfa`].
    pub state_index: u32,
    /// The tag intended for that state.
    pub tag_intended: Tag,
}

impl<A: Ord, Tag: Ord> BuildResult<A, Tag> {
    /// Try to resolve a conflict using a hint: which `tag` should be assigned to some `input`.
    pub fn apply_hint<C, I>(&mut self, input: I, tag_intended: Tag)
                            -> ResolveError<C, Tag, I::IntoIter>
        where I: IntoIterator<Item=C>, C: Borrow<A> {
        let state_index = match self.dfa.run(0, input) {
            Err(err) => return ResolveError::InvalidInput(err),
            Ok(s) => s,
        };
        let ps = self.state_mapping[idx!(state_index)].iter().copied()
            .filter(|&p| self.position_info[idx!(p)].is_accept())
            .collect_vec();
        match ps.len() {
            0 => ResolveError::NotAcceptState(NotAcceptState { state_index, tag_intended }),
            1 => ResolveError::NoConflict(ps[0]),
            _ => {
                let k = narrow!(self.position_info.len());
                self.position_info.push(PosEntry {
                    info: PosInfo::Accept(tag_intended),
                    follow_pos: BTreeSet::default(),
                });
                self.state_mapping[idx!(state_index)] = std::iter::once(k).collect();
                ResolveError::Resolved(ps.into_boxed_slice())
            }
        }
    }

    /// Try to resolve conflicts using a series of hints.
    /// See also [`apply_hint`](BuildResult::apply_hint).
    pub fn apply_hints<C, I, H>(&mut self, hints: H) -> Vec<ResolveError<C, Tag, I::IntoIter>>
        where I: IntoIterator<Item=C>, H: IntoIterator<Item=(I, Tag)>, C: Borrow<A> {
        hints.into_iter().map(|hint| self.apply_hint(hint.0, hint.1)).collect()
    }

    /// Resolve all the conflicts and produce the final DFA.
    pub fn try_resolve(self) -> Result<Resolved<A, Tag>, (Self, Vec<Conflict>)>
        where A: Clone, Tag: Clone {
        let mut result = BTreeMap::new();
        let mut errors = Vec::new();
        let mut reachability = None;
        for (ps, k) in self.state_mapping.iter().zip(0..) {
            let ps = ps.iter().copied()
                .filter(|&p| self.position_info[idx!(p)].is_accept())
                .collect_vec();
            match ps.len() {
                0 => {}
                1 => match &self.position_info[idx!(ps[0])].info {
                    PosInfo::Accept(tag) => { result.insert(k, tag.clone()); }
                    PosInfo::Normal(_) => unreachable!(),
                }
                _ => errors.push(Conflict {
                    interpretations: reachability.get_or_insert_with(|| self.reachability())
                        .try_get_paths_for(k, &self).unwrap()
                        .into_iter().map(Vec::into_boxed_slice).collect(),
                })
            }
        }
        if errors.is_empty() {
            Ok(Resolved {
                dfa: self.dfa,
                tags: result.into_iter().collect(),
            })
        } else {
            Err((self, errors))
        }
    }
}

impl<A: Ord + Clone> Expr<A> {
    /// Build a [`Dfa`] from a regular expression.
    /// See also [`Builder::build`].
    pub fn build(&self) -> BuildResult<A, ()> {
        let mut builder = Builder::default();
        let info = self.traverse_extended(&mut builder, ());
        builder.build(info)
    }

    /// Build a [`Dfa`] from many regular expressions.
    /// See also [`Builder::build`].
    pub fn build_many<'a, I, Tag>(exprs: I) -> BuildResult<A, Tag>
        where I: IntoIterator<Item=(&'a Expr<A>, Tag)>, A: 'a {
        let mut builder = Builder::default();
        let mut info = ExprInfo::default();
        for (expr, tag) in exprs {
            info |= expr.traverse_extended(&mut builder, tag);
        }
        builder.build(info)
    }
}
