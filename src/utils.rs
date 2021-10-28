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

//! Common utilities.

use std::iter::FromIterator;
use std::fmt::{Display, Formatter};
use std::ops::{Bound, RangeBounds};
use crate::utils::tuple::{TupleCompare, TupleRest, TupleSplit};

pub mod tuple;

/// Literals for boxed slices. Equivalent to `vec![...].into_boxed_slice()`.
///
/// ```
/// # use rowantlr::r#box;
/// assert_eq!(r#box![1, 2, 3], vec![1, 2, 3].into_boxed_slice());
/// assert_eq!(r#box![1, 2, 3, ], vec![1, 2, 3, ].into_boxed_slice());
/// assert_eq!(r#box![42; 3], vec![42; 3].into_boxed_slice());
/// ```
#[macro_export]
macro_rules! r#box {
    ($($es: expr),* $(,)?) => {
        ::std::vec![$($es),+].into_boxed_slice()
    };
    ($e: expr; $n: expr) => {
        ::std::vec![$e; $n].into_boxed_slice()
    }
}

/// Compare references as if they were raw pointers.
pub mod by_address {
    use std::cmp::Ordering;

    /// Fast-forward to [`PartialOrd`] for pointers. To be used by `Derivative`.
    #[inline(always)]
    pub fn eq<T: ?Sized>(x: &&T, y: &&T) -> bool {
        std::ptr::eq(*x, *y)
    }

    /// Fast-forward to [`PartialOrd`] for pointers. To be used by `Derivative`.
    #[inline(always)]
    pub fn partial_cmp<T: ?Sized>(x: &&T, y: &&T) -> Option<Ordering> {
        let x: *const T = *x;
        let y: *const T = *y;
        x.partial_cmp(&y)
    }

    /// Fast-forward to [`Ord`] for pointers. To be used by `Derivative`.
    #[inline(always)]
    pub fn cmp<T: ?Sized>(x: &&T, y: &&T) -> Ordering {
        let x: *const T = *x;
        let y: *const T = *y;
        x.cmp(&y)
    }
}

/// Wraps a [`DisplayDot2TeX`] type for convenient formatting.
pub struct Dot2TeX<'a, A: ?Sized, Env: ?Sized>(&'a A, &'a Env);

impl<'a, A: DisplayDot2TeX<Env> + ?Sized, Env: ?Sized> Display for Dot2TeX<'a, A, Env> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_dot2tex(self.1, f)
    }
}

/// Display the automata by going through [`dot2tex`](https://github.com/kjellmf/dot2tex/)
/// via [`dot2texi`](https://www.ctan.org/pkg/dot2texi) and then `TikZ` in `LaTeX`.
///
/// To typeset this automata, make sure `dot2tex` is in `PATH`.
pub trait DisplayDot2TeX<Env: ?Sized = ()> {
    /// Formats in `dot2tex` to the given formatter.
    fn fmt_dot2tex(&self, env: &'_ Env, f: &mut Formatter<'_>) -> std::fmt::Result;
    /// Wraps the type for convenient formatting.
    fn display_dot2tex<'a>(&'a self, env: &'a Env) -> Dot2TeX<'a, Self, Env> {
        Dot2TeX(self, env)
    }
}

/// Simple versions of the types and traits.
pub mod simple {
    use std::fmt::Formatter;

    /// Provide short-cut methods for [`DisplayDot2TeX`] with a nullary environment.
    pub trait DisplayDot2TeX: super::DisplayDot2TeX + private::Sealed {
        /// Formats in `dot2tex` to the given formatter.
        fn fmt_dot2tex_(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            self.fmt_dot2tex(&(), f)
        }
        /// Wraps the type for convenient formatting.
        fn display_dot2tex_(&self) -> super::Dot2TeX<Self, ()> {
            self.display_dot2tex(&())
        }
    }

    impl<T: super::DisplayDot2TeX> DisplayDot2TeX for T {}

    mod private {
        pub trait Sealed {}

        impl<T> Sealed for T {}
    }
}

/// Declares a set of types that are safe to display as `dot2tex` format directly via [`Display`].
#[macro_export]
macro_rules! display_dot2tex_via_display {
    ($($t: ty),+ $(,)?) => {
        $(
            impl<Env: ?Sized> DisplayDot2TeX<Env> for $t {
                fn fmt_dot2tex(&self, _: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{}", self)
                }
            }
        )+
    }
}

macro_rules! display_dot2tex_via_deref {
    ($t: ty $(where $($p: tt)+)?) => {
        impl<$($($p)+ ,)? Env: ?Sized> DisplayDot2TeX<Env> for $t {
            fn fmt_dot2tex(&self, env: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
                <<Self as ::std::ops::Deref>::Target as DisplayDot2TeX<Env>>::fmt_dot2tex(
                    ::std::ops::Deref::deref(self), env, f)
            }
        }
    }
}

display_dot2tex_via_display!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
display_dot2tex_via_deref!(String);
display_dot2tex_via_deref!(&'a T where 'a, T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(Box<T> where T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(std::rc::Rc<T> where T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(std::sync::Arc<T> where T: DisplayDot2TeX<Env> + ?Sized);
display_dot2tex_via_deref!(std::borrow::Cow<'_, B> where B: DisplayDot2TeX<Env> + ToOwned + ?Sized);

const TEX_SPECIAL: &[char] = &['#', '$', '%', '&', '\\', '^', '_', '{', '}', '~'];

impl<Env: ?Sized> DisplayDot2TeX<Env> for str {
    fn fmt_dot2tex(&self, _: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut rest = self;
        while let Some(idx) = rest.find(TEX_SPECIAL) {
            if idx != 0 { f.write_str(&rest[..idx])?; }
            let mut rest_chars = rest[idx..].chars();
            f.write_str(match rest_chars.next().unwrap() {
                '#' => r"\#",
                '$' => r"\$",
                '%' => r"\%",
                '&' => r"\&",
                '\\' => r"\backslash ",
                '^' => r"\char94 ",
                '_' => r"\_",
                '{' => r"\{",
                '}' => r"\}",
                '~' => r"\textasciitilde ",
                _ => unreachable!(),
            })?;
            rest = rest_chars.as_str();
        }
        if !rest.is_empty() { f.write_str(rest)?; }
        Ok(())
    }
}

/// More utility functions for iterators.
pub trait IterHelper: Iterator {
    /// [`Iterator::map`] followed by [`Iterator::reduce`], with lifetime issues properly handled.
    fn reduce_map<V, A>(mut self, visitor: &mut V,
                        f: impl Fn(&mut V, Self::Item) -> A,
                        g: impl Fn(&mut V, A, A) -> A) -> Option<A>
        where Self: Sized, V: ?Sized {
        let first = f(visitor, self.next()?);
        Some(self.fold(first, |res, x| {
            let a = f(visitor, x);
            g(visitor, res, a)
        }))
    }
}

impl<I: Iterator> IterHelper for I {}

/// An immutable, flat dictionary.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Dict<K>(Box<[K]>);

struct SingularRange<A>(A);

impl<A> RangeBounds<A> for SingularRange<A> {
    fn start_bound(&self) -> Bound<&A> { Bound::Included(&self.0) }
    fn end_bound(&self) -> Bound<&A> { Bound::Included(&self.0) }
    fn contains<U>(&self, item: &U) -> bool
        where A: PartialOrd<U>, U: ?Sized + PartialOrd<A> {
        self.0 == *item
    }
}

/// Types related to [`Dict`].
pub mod dict {
    use crate::utils::tuple::TupleRest;

    /// Double-ended iterator
    pub type Iter<'a, K, Q> = std::iter::Map<
        std::slice::Iter<'a, K>,
        fn(&'a K) -> <K as TupleRest<'a, Q>>::Rest
    >;
}

impl<K> Dict<K> {
    fn locate<Q>(&self, key: &Q) -> Result<usize, usize>
        where Q: ?Sized, K: TupleCompare<Q> {
        self.0.binary_search_by(|a| a.tuple_compare(key))
    }

    fn partition_point<F>(&self, pred: F) -> usize
        where F: FnMut(&K) -> bool {
        self.0.partition_point(pred)
    }

    fn locate_lower_bound<Q>(&self, bound: Bound<&Q>) -> usize
        where Q: ?Sized, K: TupleCompare<Q> {
        match bound {
            Bound::Included(x) => self.partition_point(|k| k.tuple_compare(x).is_lt()),
            Bound::Excluded(x) => self.partition_point(|k| k.tuple_compare(x).is_le()),
            Bound::Unbounded => 0,
        }
    }

    fn locate_upper_bound<Q>(&self, bound: Bound<&Q>) -> usize
        where Q: ?Sized, K: TupleCompare<Q> {
        match bound {
            Bound::Included(x) => self.partition_point(|k| k.tuple_compare(x).is_le()),
            Bound::Excluded(x) => self.partition_point(|k| k.tuple_compare(x).is_lt()),
            Bound::Unbounded => self.0.len(),
        }
    }

    /// Returns a reference to the slice in this dict.
    pub fn as_slice(&self) -> &[K] { &self.0 }

    /// Returns `true` if the dict contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
        where Q: ?Sized, K: TupleCompare<Q> {
        self.locate(key).is_ok()
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get<'a, Q>(&'a self, key: Q) -> Option<K::Rest>
        where K: TupleCompare<Q>, K: TupleRest<'a, Q> {
        let p = self.locate(&key).ok()?;
        Some(self.0[p].borrow_rest())
    }

    /// Returns the key-value pair corresponding to the supplied key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<&K>
        where K: TupleCompare<Q> {
        let p = self.locate(key).ok()?;
        Some(&self.0[p])
    }

    /// Creates a consuming iterator visiting all the entries, in sorted order. The map cannot be
    /// used after calling this. The iterator element type is `K`.
    pub fn into_raw(self) -> Box<[K]> { self.0 }

    /// Creates a consuming iterator visiting all the keys, in sorted order. The map cannot be
    /// used after calling this. The iterator element type is `K`.
    pub fn into_keys<const N: usize>(self) -> impl Iterator<Item=K::Init>
        where K: TupleSplit<N> {
        self.0.into_vec().into_iter().map(K::tuple_split).map(|(k, _)| k)
    }

    /// Creates a consuming iterator visiting all the values, in order by key. The map cannot be
    /// used after calling this. The iterator element type is `V`.
    pub fn into_values<const N: usize>(self) -> impl Iterator<Item=K::Tail>
        where K: TupleSplit<N> {
        self.0.into_vec().into_iter().map(K::tuple_split).map(|(_, v)| v)
    }

    /// Returns `true` if the dict contains no elements.
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    /// Gets an iterator over the entries of the map, sorted by key.
    pub fn iter(&self) -> std::slice::Iter<K> { self.0.iter() }

    fn indices_range<Q, R>(&self, range: R) -> (usize, usize)
        where Q: ?Sized, K: TupleCompare<Q>, R: RangeBounds<Q> {
        let l = self.locate_lower_bound(range.start_bound());
        let r = self.locate_upper_bound(range.end_bound());
        (l, r)
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the dict. The simplest
    /// way is to use the range syntax `min..max`, thus `range(min..max)` will yield elements from
    /// `min` (inclusive) to `max` (exclusive). The range may also be entered as
    /// `(Bound<T>, Bound<T>)`, so for example `range((Excluded(4), Included(10)))` will yield a
    /// left-exclusive, right-inclusive range from `4` to `10`.
    pub fn range<'a, Q, R>(&'a self, range: R) -> dict::Iter<'a, K, Q>
        where Q: ?Sized, R: RangeBounds<Q>, K: TupleCompare<Q>, K: TupleRest<'a, Q> {
        let (l, r) = self.indices_range(range);
        self.0[l..r].iter().map(K::borrow_rest)
    }

    /// Constructs a double-ended iterator for some specific key in the dict.
    pub fn equal_range<'a, Q>(&'a self, q: Q) -> dict::Iter<'a, K, Q>
        where K: TupleCompare<Q>, K: TupleRest<'a, Q> {
        self.range(SingularRange(q))
    }
}

impl<K: Ord> From<Vec<K>> for Dict<K> {
    fn from(buffer: Vec<K>) -> Self {
        Dict::from(buffer.into_boxed_slice())
    }
}

impl<K: Ord> From<Box<[K]>> for Dict<K> {
    fn from(mut buffer: Box<[K]>) -> Self {
        buffer.sort_unstable();
        Dict(buffer)
    }
}

impl<K: Ord> FromIterator<K> for Dict<K> {
    fn from_iter<T: IntoIterator<Item=K>>(iter: T) -> Self {
        let vec = Vec::from_iter(iter);
        Dict::from(vec)
    }
}

impl<K> IntoIterator for Dict<K> {
    type Item = K;
    type IntoIter = std::vec::IntoIter<K>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_vec().into_iter()
    }
}

impl<'a, K> IntoIterator for &'a Dict<K> {
    type Item = &'a K;
    type IntoIter = std::slice::Iter<'a, K>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
