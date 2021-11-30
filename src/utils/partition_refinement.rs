/*
 * rowantlr: ANTLR-like parser generator framework targeting rowan
 * Copyright (C) 2021  Xie Ruifeng
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

//! The [partition refinement](https://en.wikipedia.org/wiki/Partition_refinement) algorithm.
//!
//! The algorithm is implemented by maintaining the following information:
//! - all the sets, associated with each set `S` its collection of elements.
//! - all the elements, associated with each element the set it belongs to.
//!
//! These requirements are captured in structure [`Partitions`] and trait [`IndexManager`]. For most
//! cases, the `n` elements are representable by `0..n`, and thus [`TrivialIdxMan`] can be used.
//!
//! # Example
//!
//! ```
//! # use rowantlr::utils::partition_refinement::{Partitions, TrivialIdxMan};
//! # fn assert_equal<'a, I, const N: usize>(a: I, b: [&'a [u64]; N])
//! #     where I: IntoIterator, I::Item: std::fmt::Debug + PartialEq<&'a [u64]> {
//! #     itertools::assert_equal(a, b)
//! # }
//! let mut partitions = Partitions::<u64>::new(vec![0, 1, 2, 3, 4, 5, 6]);
//! partitions.refine_with([1, 3, 5], &mut ()); {
//!     assert_eq!(partitions.parts().len(), 2);
//!     assert_equal(partitions.parts(), [&[0, 6, 2, 4], &[5, 3, 1]]);
//! }
//! partitions.refine_with([4, 5, 6], &mut ()); {
//!     assert_eq!(partitions.parts().len(), 4);
//!     assert_equal(partitions.parts(), [&[0, 2], &[1, 3], &[6, 4], &[5]]);
//! }
//! partitions.refine_with([0, 2, 4], &mut ()); {
//!     assert_eq!(partitions.parts().len(), 5);
//!     assert_equal(partitions.parts(), [&[2, 0], &[1, 3], &[6], &[5], &[4]]);
//! }
//! // random access to parts is also possible
//! assert_eq!(partitions.part(1), &[1, 3]);
//! assert_eq!(partitions.part(4), &[4]);
//! ```
//!
//! Use [`Intervals`] to collect all newly-generated parts during the refinement:
//! ```
//! # use itertools::assert_equal;
//! # use rowantlr::utils::interval::Intervals;
//! # use rowantlr::utils::partition_refinement::Partitions;
//! let mut partitions = Partitions::<u64>::new_trivial(7);
//! let mut intervals = Intervals::new();
//! partitions.refine_with([1, 3, 5], &mut intervals); { // 0 6 2 4 | 5 3 1
//!     assert_equal(&intervals, &[4..7]);
//!     assert_equal(intervals.parts(&partitions).into_elements(), [&[5, 3, 1]]);
//! }
//! partitions.refine_with([4, 5, 6], &mut intervals); { // 0 2 | 6 4 | 1 3 | 5
//!     assert_equal(&intervals, &[0..2, 4..7]);
//!     assert_equal(intervals.parts(&partitions).into_elements(),
//!                  [&[1_u64, 3] as &[u64], &[5], &[0, 2]]);
//! }
//! partitions.refine_with([0, 2, 4], &mut intervals); { // 2 0 | 6 | 4 | 1 3 | 5
//!     assert_equal(&intervals, &[0..3, 4..7]);
//!     assert_equal(intervals.parts(&partitions).into_elements(),
//!                  [&[1_u64, 3] as &[u64], &[5], &[2, 0], &[6]]);
//! }
//! ```
//! Note that the intervals are only meaningful when interpreted together with the partitions.
//! These intervals are only exposed for debug purposes.

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::ops::{Index, Range};
use super::interval::{Interval, Intervals};

/// Maintain the reverse mapping from elements in a [`slice`] to their indices.
pub trait IndexManager<Element> {
    /// Initialise from some specific slice.
    fn from_slice(buffer: &[Element]) -> Self;
    /// Get the index of some `element`.
    fn index_of(&self, element: &Element) -> usize;
    /// Swap the indices of two elements.
    fn swap_index(&mut self, x: &Element, y: &Element);
}

/// A trivial [`IndexManager`] for bounded integral elements (essentially `0..n`).
///
/// ```
/// # use rowantlr::utils::partition_refinement::{TrivialIdxMan, IndexManager};
/// let mut p = TrivialIdxMan::from_slice(&[1_u64, 2, 0]);
/// assert_eq!(p.index_of(&1_u64), 0);
/// assert_eq!(p.index_of(&0_u64), 2);
/// p.swap_index(&1_u64, &0_u64);
/// assert_eq!(p.index_of(&1_u64), 2);
/// assert_eq!(p.index_of(&0_u64), 0);
/// ```
///
/// The slice must consist exactly of `0..n` (in any order), or [`from_slice`] will panic:
///
/// ```should_panic
/// # use rowantlr::utils::partition_refinement::{TrivialIdxMan, IndexManager};
/// let _ = TrivialIdxMan::from_slice(&[1_u64, 2, 3]); // panic
/// ```
///
/// [`from_slice`]: TrivialIdxMan::from_slice
#[derive(Debug, Eq, PartialEq)]
pub struct TrivialIdxMan<E>(Box<[E]>);

impl<E> IndexManager<E> for TrivialIdxMan<E>
    where E: Copy + TryInto<usize>, E::Error: std::fmt::Debug,
          usize: TryInto<E>, <usize as TryInto<E>>::Error: std::fmt::Display {
    fn from_slice(buffer: &[E]) -> Self {
        let mut result = vec![narrow!(buffer.len()); buffer.len()];
        for (n, x) in buffer.iter().enumerate() {
            result[idx!(*x)] = narrow!(n);
        }
        assert!(result.iter().all(|&i| idx!(i) != buffer.len()),
                "elements for TrivialIdxMan should be exactly 0..n");
        TrivialIdxMan(result.into_boxed_slice())
    }
    fn index_of(&self, element: &E) -> usize {
        idx!(self.0[idx!(*element)])
    }
    fn swap_index(&mut self, x: &E, y: &E) {
        self.0.swap(idx!(*x), idx!(*y));
    }
}

/// A sub set of elements in partition refinement. Never invalidated after generation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Part {
    start: u32,
    end: u32,
}

impl Part {
    fn new(start: u32, end: u32) -> Self {
        debug_assert!(start <= end, "invalid range for Part");
        Part { start, end }
    }
    fn len(self) -> u32 { self.end - self.start }
    fn as_range(self) -> Range<usize> { idx!(self.start)..idx!(self.end) }
    fn as_interval(self) -> Interval { Interval::from(self.start..self.end) }
    pub(super) fn from_interval(interval: Interval) -> Self {
        Self::new(interval.start, interval.end)
    }
}

/// Perform partition refinement on some elements.
#[derive(Debug, Eq, PartialEq)]
pub struct Partitions<E, P = TrivialIdxMan<E>> {
    back_buffer: Vec<E>,
    parent_part: Vec<u32>,
    partitions: Vec<Part>,
    positions: P,
}

impl<E> Partitions<E, TrivialIdxMan<E>> {
    /// Initialise for partition refinement with the [`TrivialIdxMan`].
    ///
    /// Equivalent to [`Partitions::new`], but might help with efficiency.
    /// ```
    /// # use rowantlr::utils::partition_refinement::Partitions;
    /// assert_eq!(Partitions::new_trivial(6), Partitions::<u64>::new((0..6).collect()));
    /// ```
    pub fn new_trivial(count: E) -> Partitions<E>
        where E: Copy + TryInto<usize>, E::Error: std::fmt::Debug,
              usize: TryInto<E>, <usize as TryInto<E>>::Error: std::fmt::Display {
        let back_buffer = (0..idx!(count)).map(|i| narrow!(i)).collect();
        let count = idx!(count);
        Partitions {
            back_buffer,
            parent_part: vec![0; count],
            partitions: vec![Part::new(0, narrow!(count => u32))],
            positions: TrivialIdxMan((0..count).map(|i| narrow!(i)).collect()),
        }
    }
}

/// [`Part`] managers controls the behaviour of [`Partitions::refine_with`] when new parts are
/// generated during the refinement. For a "do-nothing" manager, use `()`.
pub trait PartManager {
    /// A summary of all new parts.
    type Summary;
    /// Generate the summary.
    fn gen_summary(&mut self) -> Self::Summary;
    /// New parts are always generated in pairs.
    fn new_part_formed(&mut self, x: Part, y: Part);
}

impl PartManager for () {
    type Summary = ();
    fn gen_summary(&mut self) -> Self::Summary {}
    fn new_part_formed(&mut self, _: Part, _: Part) {}
}

impl PartManager for Intervals {
    type Summary = ();
    fn gen_summary(&mut self) -> Self::Summary {}
    fn new_part_formed(&mut self, x: Part, y: Part) {
        let p = std::cmp::min_by_key(x, y, |p| p.len());
        self.insert(p.as_interval());
    }
}

impl<E, P: IndexManager<E>> Partitions<E, P> {
    /// Initialise for partition refinement.
    pub fn new(elements: Vec<E>) -> Partitions<E, P> {
        Partitions {
            parent_part: vec![0; elements.len()],
            partitions: vec![Part::new(0, narrow!(elements.len()))],
            positions: P::from_slice(&elements),
            back_buffer: elements,
        }
    }

    /// Get the partition index of some specific element.
    pub fn parent_part_of(&self, x: &E) -> u32 {
        self.parent_part[self.position_of(x)]
    }

    fn position_of(&self, x: &E) -> usize {
        self.positions.index_of(x)
    }

    /// Get a slice for some partition.
    pub fn part(&self, n: u32) -> &[E] {
        let p = self.partitions[idx!(n)];
        &self.back_buffer[p.as_range()]
    }

    /// Get all the partitions.
    pub fn parts(&self) -> impl Iterator<Item=&[E]> + ExactSizeIterator {
        self.partitions.iter().map(|rng| &self.back_buffer[rng.as_range()])
    }

    /// Promote a part (determined by its index `n`) to index 0.
    pub fn promote_to_head(&mut self, n: u32) {
        if n == 0 { return; }
        self.partitions.swap(0, idx!(n));
        for p in [0, n] {
            for parent in &mut self.parent_part[self.partitions[idx!(p)].as_range()] {
                *parent = p;
            }
        }
    }

    pub(super) fn split_interval(&self, interval: &mut Interval) -> Interval {
        let p = self.partitions[idx!(self.parent_part[idx!(interval.start)])];
        assert_eq!(p.start, interval.start, "the part is not properly aligned");
        interval.start = p.end;
        p.as_interval()
    }

    /// Refine with a set of (non-duplicated) elements typed `E`.
    ///
    /// Beware that if an element appears twice in `s`, the answer will be incorrect, the whole
    /// data structure may be corrupted, and the program may or may not panic.
    pub fn refine_with<A, I, M>(&mut self, s: I, manager: &mut M) -> M::Summary
        where A: Borrow<E>, I: IntoIterator<Item=A>, M: PartManager {
        let mut affected = BTreeMap::new();
        for x in s {
            let x = x.borrow();
            let parent = self.parent_part_of(x);
            // record 'parent' set is affected
            let n = affected.entry(parent).or_insert(0);
            *n += 1;
            // swap 'x' out of 'parent'
            let last = self.partitions[idx!(parent)].end - *n;
            let pos = self.position_of(x);
            self.positions.swap_index(x, &self.back_buffer[idx!(last)]);
            self.back_buffer.swap(pos, idx!(last));
        }
        for (a, n_moved) in affected {
            let a = idx!(a);
            assert!(n_moved <= self.partitions[a].len());
            // all of the elements have been moved out from set 'a'
            if n_moved == self.partitions[a].len() { continue; }
            // form a new set 'new_a'
            let new_a = narrow!(self.partitions.len());
            let new_a_end = self.partitions[a].end;
            let new_a_begin = new_a_end - n_moved;
            let new_a_rng = Part::new(new_a_begin, new_a_end);
            self.partitions.push(new_a_rng);
            // shrink the parent set 'a'
            self.partitions[a].end = new_a_begin;
            // record set pair '(a, new_a)'
            let a_rng = self.partitions[a];
            manager.new_part_formed(a_rng, new_a_rng);
            // update parents for elements in 'new_a'
            new_a_rng.as_range().for_each(|p| self.parent_part[p] = new_a);
        }
        manager.gen_summary()
    }
}

impl<E, P> Index<Part> for Partitions<E, P> {
    type Output = [E];
    fn index(&self, index: Part) -> &[E] {
        &self.back_buffer[index.as_range()]
    }
}
