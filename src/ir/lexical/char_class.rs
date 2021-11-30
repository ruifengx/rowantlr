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

//! Character class support for regular expressions.

use std::collections::BTreeSet;
use std::fmt::Formatter;
use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use itertools::Itertools;
use crate::utils::Dict;
use crate::utils::partition_refinement::Partitions;
use super::{Expr, Op};

/// Wrapper for [`u32`] as characters and off-by-one characters. [`CharClass`]es are left-closed
/// right-open intervals, so right end points of the intervals are not necessarily valid [`char`]s.
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct Char(pub u32);

impl std::fmt::Debug for Char {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match char::from_u32(self.0) {
            None => write!(f, "#{:04x}", self.0),
            Some(c) => write!(f, "'{}'", c),
        }
    }
}

impl From<Char> for usize {
    fn from(c: Char) -> usize { c.0 as usize }
}

impl From<u32> for Char {
    fn from(c: u32) -> Char { Char(c) }
}

impl From<char> for Char {
    fn from(c: char) -> Char { Char(c as u32) }
}

impl From<Char> for u32 {
    fn from(c: Char) -> u32 { c.0 }
}

impl TryFrom<Char> for char {
    type Error = <char as TryFrom<u32>>::Error;
    fn try_from(c: Char) -> Result<char, Self::Error> {
        char::try_from(c.0)
    }
}

/// Character range.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct CharClass {
    /// Lower bound of this character class (inclusive).
    pub start: Char,
    /// Upper bound of this character class (exclusive).
    pub end: Char,
}

impl From<char> for CharClass {
    fn from(c: char) -> Self {
        CharClass::from_range((Bound::Included(c), Bound::Included(c)))
    }
}

impl PartialEq<char> for CharClass {
    fn eq(&self, &c: &char) -> bool {
        self.start.0 == c as u32
            && self.end.0 == c as u32 + 1
    }
}

impl CharClass {
    #[inline(always)]
    fn from_range<R>(range: R) -> Self
        where R: RangeBounds<char> {
        let start = Char(match range.start_bound() {
            Bound::Included(&start) => start as u32,
            Bound::Excluded(&start) => start as u32 + 1,
            Bound::Unbounded => u32::MIN,
        });
        let end = Char(match range.end_bound() {
            Bound::Included(&end) => end as u32 + 1,
            Bound::Excluded(&end) => end as u32,
            Bound::Unbounded => char::MAX as u32 + 1,
        });
        CharClass { start, end }
    }
}

macro_rules! char_class_from_range {
    ($($range: ty),+ $(,)?) => {
        $(
            impl From<$range> for CharClass {
                fn from(range: $range) -> Self {
                    CharClass::from_range(range)
                }
            }
            impl PartialEq<$range> for CharClass {
                fn eq(&self, range: &$range) -> bool {
                    #![allow(clippy::cmp_owned)]
                    *self == CharClass::from(range.clone())
                }
            }
        )+
    }
}

char_class_from_range! {
    Range<char>,
    RangeInclusive<char>,
    RangeTo<char>,
    RangeToInclusive<char>,
    RangeFrom<char>,
    RangeFull,
}

impl Expr<CharClass> {
    fn collect_inputs<'a, I>(exprs: I) -> Vec<(Char, u32)>
        where I: IntoIterator<Item=&'a Expr<CharClass>> {
        fn visit(expr: &Expr<CharClass>, inputs: &mut BTreeSet<Char>) {
            match &expr.0 {
                Op::Singleton(x) => {
                    inputs.insert(x.start);
                    inputs.insert(x.end);
                }
                Op::Union(xs) => xs.iter().for_each(|x| visit(x, inputs)),
                Op::Concat(xs) => xs.iter().for_each(|x| visit(x, inputs)),
                Op::Some(x) => visit(x, inputs),
            }
        }
        let mut inputs = BTreeSet::new();
        inputs.insert(Char::from('\0'));
        exprs.into_iter().for_each(|expr| visit(expr, &mut inputs));
        inputs.into_iter().zip(0..).collect()
    }

    /// Analyse the [`CharClass`]es, and produce a character classifier and an [`Expr`] with char
    /// classes replaced by their indices in the classifier.
    ///
    /// ```
    /// # use itertools::assert_equal;
    /// # use rowantlr::ir::lexical::{Expr, char_class::CharClass};
    /// use rowantlr::ir::lexical::char_class::Char;
    /// # use rowantlr::utils::Dict;
    /// // auxiliary function for constructing a singleton of a char class:
    /// fn char_class<C: Into<CharClass>>(c: C) -> Expr<CharClass> { Expr::singleton(c.into()) }
    /// // a fictitious lexical grammar for identifiers
    /// let mod_id_start = Expr::union([
    ///     char_class('A'..='Z'),
    ///     char_class('$'),
    /// ]);
    /// let var_id_start = Expr::union([
    ///     char_class('a'..='z'),
    ///     char_class('_'),
    /// ]);
    /// let id_cont = Expr::many(Expr::union([
    ///     char_class('a'..='z'),
    ///     char_class('A'..='Z'),
    ///     char_class('_'),
    ///     char_class('0'..='9'),
    ///     char_class('\''),
    /// ]));
    /// let mod_id = Expr::concat([mod_id_start, id_cont.clone()]);
    /// let var_id = Expr::concat([var_id_start, id_cont]);
    /// let (ids, inputs) = Expr::freeze_char_class([&mod_id, &var_id]);
    /// // the expressions with input frozen
    /// assert_equal(ids, [
    ///     Expr::concat([
    ///         Expr::union_of([1, 3]),
    ///         Expr::many(Expr::union_of([2, 3, 4])),
    ///     ]),
    ///     Expr::concat([
    ///         Expr::singleton(4),
    ///         Expr::many(Expr::union_of([2, 3, 4]))
    ///     ]),
    /// ]);
    /// // auxiliary function for finding the next char:
    /// fn next_char(c: char) -> char { char::from_u32(c as u32 + 1).unwrap() }
    /// // the input classifier
    /// assert_eq!(inputs, [
    ///     ('\0', 0),
    ///     ('$',  1), (next_char('$'),  0),
    ///     ('\'', 2), (next_char('\''), 0),
    ///     ('0',  2), (next_char('9'),  0),
    ///     ('A',  3), (next_char('Z'),  0),
    ///     ('_',  4), (next_char('_'),  0),
    ///     ('a',  4), (next_char('z'),  0),
    /// ].into_iter().map(|(c, k)| (Char::from(c), k)).collect());
    /// ```
    /// For usage of the resulting classifier, see [`Dict::classify`].
    pub fn freeze_char_class<'a, I>(exprs: I) -> (Vec<Expr<u32>>, Dict<(Char, u32)>)
        where I: IntoIterator<Item=&'a Expr<CharClass>>, I::IntoIter: Clone {
        let exprs = exprs.into_iter();
        let inputs = Dict::from(Expr::collect_inputs(exprs.clone()));
        let mut partitions = Partitions::new_trivial(narrow!(inputs.len() => u32));

        #[must_use]
        struct Collector<'a> {
            inputs: &'a Dict<(Char, u32)>,
            partitions: &'a mut Partitions<u32>,
            result: BTreeSet<u32>,
        }

        impl<'a> Drop for Collector<'a> {
            fn drop(&mut self) {
                self.partitions.refine_with(std::mem::take(&mut self.result), &mut ())
            }
        }

        impl<'a> Collector<'a> {
            fn new(inputs: &'a Dict<(Char, u32)>, partitions: &'a mut Partitions<u32>) -> Self {
                Collector { inputs, partitions, result: BTreeSet::new() }
            }

            fn restart(&mut self) -> Collector {
                Collector {
                    inputs: self.inputs,
                    partitions: self.partitions,
                    result: BTreeSet::new(),
                }
            }

            fn collect(&mut self, expr: &Expr<CharClass>) {
                match &expr.0 {
                    Op::Singleton(x) => self.result.extend(
                        self.inputs.range((&x.start, )..(&x.end, )).copied()),
                    Op::Union(xs) => xs.iter().for_each(|x| self.collect(x)),
                    Op::Concat(xs) => xs.iter().for_each(|x| self.restart().collect(x)),
                    Op::Some(x) => self.restart().collect(x),
                }
            }
        }

        exprs.clone().for_each(|expr| Collector::new(&inputs, &mut partitions).collect(expr));

        let mut inputs = inputs.into_raw().into_vec();
        inputs.iter_mut().for_each(|(_, k)| *k = partitions.parent_part_of(k));
        let inputs = inputs.into_iter()
            .group_by(|(_, k)| *k).into_iter()
            .map(|(_, mut g)| g.next().unwrap()).collect::<Dict<_>>();

        #[must_use]
        struct Transformer<'a> {
            inputs: &'a Dict<(Char, u32)>,
            current: BTreeSet<u32>,
            target: &'a mut Vec<Expr<u32>>,
        }

        impl<'a> Drop for Transformer<'a> {
            fn drop(&mut self) {
                if !self.current.is_empty() {
                    let last = std::mem::take(&mut self.current);
                    self.target.push(Expr::union(last.into_iter().map(Expr::singleton)));
                }
            }
        }

        impl<'a> Transformer<'a> {
            fn go(inputs: &'a Dict<(Char, u32)>, expr: &Expr<CharClass>) -> Expr<u32> {
                let mut target = Vec::new();
                Transformer { inputs, current: BTreeSet::new(), target: &mut target }.transform(expr);
                assert_eq!(target.len(), 1, "only one output expected");
                target.into_iter().next().unwrap()
            }

            // restart marks the boundaries where contents of 'current' should be separated
            fn restart(&mut self) -> Transformer {
                Transformer {
                    inputs: self.inputs,
                    current: BTreeSet::new(),
                    target: self.target,
                }
            }

            // retarget marks the boundaries where the 'target' 'Expr' sequence should be separated
            fn retarget<C, F>(&mut self, cons: C, f: F)
                where C: FnOnce(Vec<Expr<u32>>) -> Expr<u32>,
                      F: FnOnce(&mut Transformer) {
                let mut target = Vec::new();
                f(&mut Transformer {
                    inputs: self.inputs,
                    current: BTreeSet::new(),
                    target: &mut target,
                });
                self.target.push(cons(target))
            }

            fn transform(&mut self, expr: &Expr<CharClass>) {
                fn some1(xs: Vec<Expr<u32>>) -> Expr<u32> {
                    assert_eq!(xs.len(), 1, "only one child expected for Op::Some");
                    Expr::some(xs.into_iter().next().unwrap())
                }
                match &expr.0 {
                    Op::Singleton(x) => self.current.extend(
                        self.inputs.range((&x.start, )..(&x.end, )).copied()),
                    Op::Union(xs) => self.retarget(Expr::union, |t|
                        xs.iter().for_each(|x| t.transform(x))),
                    Op::Concat(xs) => self.retarget(Expr::concat, |t|
                        xs.iter().for_each(|x| t.restart().transform(x))),
                    Op::Some(x) => self.retarget(some1, |t| t.restart().transform(x)),
                }
            }
        }

        (exprs.map(|expr| Transformer::go(&inputs, expr)).collect(), inputs)
    }
}
