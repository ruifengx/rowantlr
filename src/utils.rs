/*
 * rowantlr: ANTLR-like parser generator framework targetting rowan
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

use std::fmt::{Display, Formatter};

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
