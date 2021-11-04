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
//! # fn assert_equal<'a, I, const N: usize>(a: I, b: [&'a [usize]; N])
//! #     where I: IntoIterator, I::Item: std::fmt::Debug + PartialEq<&'a [usize]> {
//! #     itertools::assert_equal(a, b)
//! # }
//! let mut partitions = Partitions::<usize, TrivialIdxMan>::new(vec![0, 1, 2, 3, 4, 5, 6]);
//! partitions.refine_with([1, 3, 5]); {
//!     assert_eq!(partitions.parts().len(), 2);
//!     assert_equal(partitions.parts(), [&[0, 6, 2, 4], &[5, 3, 1]]);
//! }
//! partitions.refine_with([4, 5, 6]); {
//!     assert_eq!(partitions.parts().len(), 4);
//!     assert_equal(partitions.parts(), [&[0, 2], &[1, 3], &[6, 4], &[5]]);
//! }
//! partitions.refine_with([0, 2, 4]); {
//!     assert_eq!(partitions.parts().len(), 5);
//!     assert_equal(partitions.parts(), [&[2, 0], &[1, 3], &[6], &[5], &[4]]);
//! }
//! ```

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::ops::Range;

/// Maintain the reverse mapping from elements in a [`slice`] to their indices.
pub trait IndexManager<Element> {
    /// Initialise from some specific slice.
    fn from_slice(buffer: &[Element]) -> Self;
    /// Get the index of some `element`.
    fn index_of(&self, element: &Element) -> usize;
    /// Swap the indices of two elements.
    fn swap_index(&mut self, x: &Element, y: &Element);
}

/// A trivial [`IndexManager`]: `Element` is essentially `0..n`.
pub struct TrivialIdxMan(Box<[usize]>);

impl<E> IndexManager<E> for TrivialIdxMan
    where E: Copy + Into<usize> {
    fn from_slice(buffer: &[E]) -> Self {
        let mut result = vec![buffer.len(); buffer.len()];
        for (n, x) in buffer.iter().enumerate() {
            result[E::into(*x)] = n;
        }
        assert!(result.iter().all(|&i| i != buffer.len()),
                "elements for TrivialIdxMan should be exactly 0..n");
        TrivialIdxMan(result.into_boxed_slice())
    }
    fn index_of(&self, element: &E) -> usize {
        self.0[E::into(*element)]
    }
    fn swap_index(&mut self, x: &E, y: &E) {
        self.0.swap(E::into(*x), E::into(*y));
    }
}

/// Perform partition refinement on some elements.
pub struct Partitions<E, P> {
    back_buffer: Vec<E>,
    parent_part: Vec<usize>,
    partitions: Vec<Range<usize>>,
    positions: P,
}

impl<E, P: IndexManager<E>> Partitions<E, P> {
    /// Initialise for partition refinement.
    pub fn new(elements: Vec<E>) -> Partitions<E, P> {
        Partitions {
            parent_part: vec![0; elements.len()],
            partitions: vec![0..elements.len()],
            positions: P::from_slice(&elements),
            back_buffer: elements,
        }
    }

    /// Get the partition index of some specific element.
    pub fn parent_part_of(&self, x: &E) -> usize {
        self.parent_part[self.position_of(x)]
    }

    fn position_of(&self, x: &E) -> usize {
        self.positions.index_of(x)
    }

    /// Get a slice for some partition.
    pub fn part(&self, n: usize) -> &[E] {
        &self.back_buffer[self.partitions[n].clone()]
    }

    /// Get all the partitions.
    pub fn parts(&self) -> impl Iterator<Item=&[E]> + ExactSizeIterator {
        self.partitions.iter().cloned().map(|rng| &self.back_buffer[rng])
    }

    /// Refine with a set of (non-duplicated) elements typed `E`.
    ///
    /// Beware that if an element appears twice in `s`, the answer will be incorrect, the whole
    /// data structure may be corrupted, and the program may or may not panic.
    pub fn refine_with<A, I>(&mut self, s: I) -> Vec<usize>
        where A: Borrow<E>, I: IntoIterator<Item=A> {
        let mut affected = BTreeMap::new();
        for x in s {
            let x = x.borrow();
            let parent = self.parent_part_of(x);
            // record 'parent' set is affected
            let n = affected.entry(parent).or_insert(0usize);
            *n += 1;
            // swap 'x' out of 'parent'
            let last = self.partitions[parent].end - *n;
            let pos = self.position_of(x);
            self.positions.swap_index(x, &self.back_buffer[last]);
            self.back_buffer.swap(pos, last);
        }
        let mut newly_formed = Vec::new();
        for (a, n_moved) in affected {
            assert!(n_moved <= self.partitions[a].len());
            // all of the elements have been moved out from set 'a'
            if n_moved == self.partitions[a].len() { continue; }
            // form a new set 'new_a'
            let new_a = self.partitions.len();
            let new_a_end = self.partitions[a].end;
            let new_a_begin = new_a_end - n_moved;
            let new_a_rng = new_a_begin..new_a_end;
            self.partitions.push(new_a_rng.clone());
            // shrink the parent set 'a'
            self.partitions[a].end = new_a_begin;
            // record set pair '(a, new_a)' if a != {}
            let a_rng = self.partitions[a].clone();
            if a_rng.len() > new_a_rng.len() {
                newly_formed.push(new_a);
            } else {
                newly_formed.push(a);
            }
            // update parents for elements in 'new_a'
            new_a_rng.for_each(|p| self.parent_part[p] = new_a);
        }
        newly_formed
    }
}
