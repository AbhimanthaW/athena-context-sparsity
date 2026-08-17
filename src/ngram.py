from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection


class NGramModel:
    """
    Fixed-order word-level n-gram language model using add-alpha smoothing.

    Model order:
        n = 1 -> unigram
        n = 2 -> bigram
        n = 3 -> trigram
        ...

    History length:
        k = n - 1
    """

    def __init__(
        self,
        n: int,
        alpha: float,
        vocabulary: dict | Collection,
    ):
        # -----------------------------------------------------------------
        # Validate model configuration
        # -----------------------------------------------------------------
        if n < 1:
            raise ValueError("n must be at least 1.")

        # Our experiment uses strictly positive additive smoothing.
        #
        # alpha = 0 would correspond to unsmoothed MLE and would make
        # unseen histories undefined (0/0), so it is deliberately rejected.
        if alpha <= 0:
            raise ValueError(
                "alpha must be strictly positive."
            )

        if len(vocabulary) == 0:
            raise ValueError(
                "vocabulary must not be empty."
            )

        self.n = n
        self.k = n - 1
        self.alpha = alpha
        self.vocab_size = len(vocabulary)

        # ---------------------------------------------------------------
        # C(h)
        # ---------------------------------------------------------------
        # Counts how many times each history appears before a prediction
        # target in the training data.
        #
        # Example:
        #   history = (12, 47)
        # ---------------------------------------------------------------
        self.history_counts = Counter()

        # ---------------------------------------------------------------
        # C(h, w)
        # ---------------------------------------------------------------
        # Counts how many times target token w follows history h.
        #
        # Stored as:
        #   ((history tuple), target)
        # ---------------------------------------------------------------
        self.ngram_counts = Counter()

        # Total number of training prediction events.
        self.total_prediction_events = 0

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def fit(
        self,
        sequences: list[list[int]],
    ) -> "NGramModel":
        """
        Estimate C(h) and C(h,w) from block-structured training sequences.

        Each inner sequence is treated as independent for n-gram boundary
        purposes. We NEVER construct n-grams across separate block edges.
        """

        # Fitting replaces the previous model state.
        #
        # Without this reset, calling fit() twice would silently double all
        # counts, an extremely unpleasant research bug.
        self.history_counts.clear()
        self.ngram_counts.clear()
        self.total_prediction_events = 0

        for sequence in sequences:

            # target_idx begins at k because the first k tokens do not yet
            # have enough preceding context for an order-n model.
            for target_idx in range(self.k, len(sequence)):

                # Unigram special case:
                #
                # k = 0
                # history = ()
                if self.k == 0:
                    history = ()
                else:
                    history = tuple(
                        sequence[
                            target_idx - self.k:
                            target_idx
                        ]
                    )

                target = sequence[target_idx]

                self.history_counts[history] += 1
                self.ngram_counts[(history, target)] += 1

                self.total_prediction_events += 1

        return self

    # -------------------------------------------------------------------
    # Probability estimation
    # -------------------------------------------------------------------

    def probability(
        self,
        history,
        token: int,
    ) -> float:
        """
        Compute the add-alpha conditional probability:

                       C(h,w) + alpha
        q_alpha(w|h) = -------------------
                       C(h) + alpha |V|

        If an entire history is unseen:

            C(h) = 0

        then:

            q_alpha(w|h)
            = alpha / (alpha |V|)
            = 1 / |V|

        so the model falls back to a uniform vocabulary distribution.
        """

        if isinstance(history, list):
            history = tuple(history)

        if not isinstance(history, tuple):
            raise TypeError(
                "history must be a tuple or list."
            )

        # Protect against accidentally feeding a history from the wrong
        # model order.
        if len(history) != self.k:
            raise ValueError(
                f"Expected history length {self.k}, "
                f"received {len(history)}."
            )

        c_h = self.history_counts.get(
            history,
            0,
        )

        c_hw = self.ngram_counts.get(
            (history, token),
            0,
        )

        numerator = c_hw + self.alpha

        denominator = (
            c_h
            + self.alpha * self.vocab_size
        )

        # denominator is always > 0 because:
        #   alpha > 0
        #   |V| > 0
        return numerator / denominator

    # -------------------------------------------------------------------
    # Log-probability
    # -------------------------------------------------------------------

    def log_probability(
        self,
        history,
        token: int,
    ) -> float:
        """
        Return natural log q_alpha(w|h).

        Natural logarithms mean cross-entropy will later be measured in:

            nats / token
        """
        return math.log(
            self.probability(history, token)
        )