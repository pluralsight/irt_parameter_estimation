# IRT Parameter Estimation routines

This package implements parameter estimation for logistic Item Characteristic Curves (ICC) from Item Response Theory (IRT).

Maximum likelihood estimation (MLE) fitting routines for the following logistic models are implemented:
 
 * 1PL - 1 parameter logistic (Rausch) model
   * b (difficulty)
 * 2PL - 2 parameter logistic model
   * a (discrimination) and b (difficulty)
 * 3PL - 3 parameter logistic (Birnbaum) model
   * a (discrimination), b (difficulty) and c (pseudo-guessing)

In addition, fitting using the zeta/lamdba/c formulation is also implemented.
The difference here boils down to the logistic exponent.
The conversion is:

a (θ<sub>j</sub> - b) = ζ + λ θ<sub>j</sub>

```a * (theta_j - b) = zeta + lambda * theta_j```

This seemingly insignificant change has drastic effects on the convergence
properties (especially in the 2PL case, but also the 3PL case).

Many of the methods in this package are derived from work by
Frank B. Baker and Seock-Ho Kim:

_Item Response Theory: Parameter Estimation Techniques_

http://www.crcpress.com/product/isbn/9780824758257

The exception is the 3 parameter zeta/lambda/c implementation which to our
knowledge has not been derived or documented before.
For this reason, we include the mathematical derivations here:

[irt-zlc-formuation.pdf](doc/zlc-irt-formulation.pdf)

The original BASIC code that work was derived from can be downloaded here:

http://www.crcpress.com/downloads/DK2939/IRTPET.zip

The original python port of these hand-coded iterative schemes can be
found in the file baker_mle.py (imported as "baker").
These are mainly useful for comparative purposes (for instance, the 2PL
version matches 1-1 with the original routine's published values).

The main routines in this package (zlc_mle.py and abc_mle.py)
use scipy.optimize.root instead for greater efficiency and accuracy.
zlc uses a hybrid (zeta, lambda, c) formulation which makes the 2PL
and 3PL systems converge much more much stably.
This version is the one imported at the top level.
This version also includes an "abc emulation" mode where zlc is still
used internally, but automatic conversion is used so that the function
takes a/b/c as arguments.

abc (in abc_mle.py) uses the (a,b,c) formulation, which may also be
useful (try both for 3PL!).
If you want to really dig into this code, it is very informative to use
a side-by-side difference tool
(like Vimdiff, KDiff3, WinMerge, FileMerge, or Meld)
to compare abc_mle.py with zlc.mle.py.

All common utilities are found in util.py.
