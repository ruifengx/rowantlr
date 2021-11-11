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

//! Interval set, with insertion, (limited) queries and removal.
//!
//! [`insert`](Intervals::insert)ion and [`iter`](Intervals::iter)ation:
//! ```
//! # use itertools::assert_equal;
//! # use rowantlr::utils::interval::{Interval, Intervals};
//! let mut set = Intervals::default();
//! set.insert(17..21);
//! set.insert(3..7);
//! set.insert(23..32);
//! assert_equal(&set, &[3..7, 17..21, 23..32]);
//! set.insert(32..40);
//! assert_equal(&set, &[3..7, 17..21, 23..40]);
//! set.insert(20..21);
//! assert_equal(&set, &[3..7, 17..21, 23..40]);
//! set.insert(20..27);
//! assert_equal(&set, &[3..7, 17..40]);
//! set.insert(2..13);
//! assert_equal(&set, &[2..13, 17..40]);
//! set.insert(0..100);
//! assert_equal(&set, &[0..100]);
//! ```
//!
//! [`pop`](Intervals::pop) out the last interval:
//! ```
//! # use itertools::assert_equal;
//! # use rowantlr::utils::interval::{Interval, Intervals};
//! # let mut set = Intervals::default();
//! # set.insert(3..7);
//! # set.insert(17..21);
//! # set.insert(23..32);
//! assert_equal(&set, &[3..7, 17..21, 23..32]);
//! assert_eq!(set.pop(), Some(Interval { start: 23, end: 32 }));
//! assert_eq!(set.pop(), Some(Interval { start: 17, end: 21 }));
//! assert_eq!(set.pop(), Some(Interval { start: 3, end: 7 }));
//! assert_eq!(set.pop(), None);
//! ```

use std::collections::Bound;
use std::ops::{Range, RangeBounds, RangeInclusive, RangeTo, RangeToInclusive};

/// An interval. The invariant `start <= end` is always expected.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct Interval {
    /// Lower-bound (inclusive).
    pub start: usize,
    /// Upper-bound (exclusive).
    pub end: usize,
}

impl Interval {
    #[inline(always)]
    fn from_range<R>(range: R) -> Self
        where R: RangeBounds<usize> {
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start + 1,
            Bound::Unbounded => usize::MIN,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end + 1,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => unreachable!(),
        };
        Interval { start, end }
    }
}

macro_rules! interval_from_range {
    ($($range: ty),+ $(,)?) => {
        $(
            impl From<$range> for Interval {
                fn from(range: $range) -> Self {
                    Interval::from_range(range)
                }
            }
            impl PartialEq<$range> for Interval {
                fn eq(&self, range: &$range) -> bool {
                    #![allow(clippy::cmp_owned)]
                    *self == Interval::from(range.clone())
                }
            }
        )+
    }
}

interval_from_range! {
    Range<usize>,
    RangeInclusive<usize>,
    RangeTo<usize>,
    RangeToInclusive<usize>,
    // RangeFrom & RangeFull are open on the right
}

impl RangeBounds<usize> for Interval {
    fn start_bound(&self) -> Bound<&usize> { Bound::Included(&self.start) }
    fn end_bound(&self) -> Bound<&usize> { Bound::Excluded(&self.end) }
}

/// Interval set.
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct Intervals(Vec<Interval>);

impl Intervals {
    /// Insert an interval and merge with other existing intervals.
    #[inline(always)]
    pub fn insert<I>(&mut self, interval: I)
        where I: Into<Interval> {
        self.insert_interval(interval.into())
    }

    fn insert_interval(&mut self, x: Interval) {
        let l = self.0.partition_point(|y| y.end < x.start);
        if l == self.0.len() { return self.0.push(x); }
        debug_assert!(x.start <= self.0[l].end);
        if x.end < self.0[l].start {
            self.0.insert(l, x);
        } else {
            self.0[l].start = std::cmp::min(self.0[l].start, x.start);
            debug_assert!(self.0[l].start <= x.end);
            let r = l + self.0[l..].partition_point(|y| y.start <= x.end);
            debug_assert!(self.0[r - 1].start <= x.end);
            if r > l {
                self.0[l].end = std::cmp::max(self.0[r - 1].end, x.end);
                self.0.drain((Bound::Excluded(l), Bound::Excluded(r)));
            }
        }
    }

    /// Obtain an iterator into the intervals.
    pub fn iter(&self) -> std::slice::Iter<Interval> { self.0.iter() }

    /// Pop the last interval from this set.
    pub fn pop(&mut self) -> Option<Interval> { self.0.pop() }
}

impl<'a> IntoIterator for &'a Intervals {
    type Item = &'a Interval;
    type IntoIter = std::slice::Iter<'a, Interval>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
