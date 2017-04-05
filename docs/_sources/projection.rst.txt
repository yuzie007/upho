Projection
==========

Chemical projection
-------------------

There may be several possible schemes for the chemical projection.

E1
^^

.. math::
    :nowrap:

    \begin{align}
        w_{\mathbf{k}, \textrm{XX}'}
        &=
        \left(
        \hat{P}^{\mathbf{k}}
        \hat{P}^{\textrm{X}}
        \tilde{\mathbf{v}}
        \right)^{\dagger}
        \left(
        \hat{P}^{\mathbf{k}}
        \hat{P}^{\textrm{X}'}
        \tilde{\mathbf{v}}
        \right)
    \end{align}

Generally this scheme causes cross terms, which can be negative.

E2
^^

.. math::
    :nowrap:

    \begin{gather}
        w_{\mathbf{k}}
        =
        \left|
        \hat{P}^{\mathbf{k}}
        \tilde{\mathbf{v}}
        \right|^{2}
        \\
        w_{\textrm{X}}
        =
        \left|
        \hat{P}^{\textrm{X}}
        \tilde{\mathbf{v}}
        \right|^{2}
        \\

        w_{\mathbf{k}, \textrm{X}}
        \equiv
        w_{\mathbf{k}}
        w_{\textrm{X}}
    \end{gather}

This scheme does not cause cross terms.
Instead, :math:`w_{\mathbf{k}, \textrm{X}'}/w_{\mathbf{k}, \textrm{X}}` does no longer depend on 
:math:`\mathbf{k}`.
