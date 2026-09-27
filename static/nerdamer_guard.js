/*
 * Budget guard for nerdamer's symbolic limits.
 *
 * nerdamer 1.1.13 applies L'Hopital's rule in an unbounded loop (Calculus.js, Limit.divide), so a
 * limit such as limit(abs(x)/x, x, Infinity) never returns and freezes the browser tab: the
 * derivative of abs(x) is abs(x)/x, which gives back the same quotient on every pass.
 *
 * window.MatHudGuardedNerdamer(expression, maxSteps, maxMs) evaluates a nerdamer expression while
 * every derivative and recursive limit step taken inside it counts against a budget (a number of
 * steps and a time limit). When the budget runs out the next step throws, nerdamer's limit code
 * catches that and returns the limit unevaluated, and the result reports exceeded: true. The
 * result is {text, exceeded, error, guarded}, with error '' when nerdamer raised nothing.
 * Outside a guarded call nerdamer behaves exactly as before.
 */
(function (root) {
    'use strict';

    var DEFAULT_MAX_STEPS = 2000;
    var DEFAULT_MAX_MS = 1500;
    var BUDGET_MESSAGE = 'MatHud: symbolic computation budget exceeded';

    var budget = { active: false, steps: 0, maxSteps: 0, deadline: 0, exceeded: false };

    function spend() {
        if (!budget.active) {
            return;
        }
        budget.steps += 1;
        if (budget.exceeded || budget.steps > budget.maxSteps || Date.now() > budget.deadline) {
            budget.exceeded = true;
            throw new Error(BUDGET_MESSAGE);
        }
    }

    function wrap(owner, key) {
        var original = owner[key];
        if (typeof original !== 'function' || original.__mathudGuarded) {
            return false;
        }
        var guarded = function () {
            spend();
            return original.apply(this, arguments);
        };
        guarded.__mathudGuarded = true;
        owner[key] = guarded;
        return true;
    }

    function install(nerdamer) {
        if (!nerdamer || typeof nerdamer.getCore !== 'function') {
            return false;
        }
        var calculus = nerdamer.getCore().Calculus;
        if (!calculus || !calculus.Limit) {
            return false;
        }
        // Limit.divide and the recursive limit calls look these up as properties on every step.
        wrap(calculus, 'diff');
        wrap(calculus.Limit, 'limit');
        wrap(calculus.Limit, 'divide');
        return true;
    }

    function guardedEvaluate(expression, maxSteps, maxMs) {
        var installed = install(root.nerdamer);
        // Empty strings rather than null: Brython does not map a JavaScript null to None.
        var result = { text: '', exceeded: false, error: '', guarded: installed };
        if (budget.active) {
            // Nested call: share the outer budget.
            result.text = String(root.nerdamer(expression).text());
            return result;
        }
        budget.active = true;
        budget.steps = 0;
        budget.maxSteps = maxSteps || DEFAULT_MAX_STEPS;
        budget.deadline = Date.now() + (maxMs || DEFAULT_MAX_MS);
        budget.exceeded = false;
        try {
            result.text = String(root.nerdamer(expression).text());
        } catch (error) {
            result.error = String((error && error.message) || error);
        } finally {
            result.exceeded = budget.exceeded;
            budget.active = false;
        }
        return result;
    }

    install(root.nerdamer);
    root.MatHudGuardedNerdamer = guardedEvaluate;
    root.MatHudNerdamerGuardBudgetMessage = BUDGET_MESSAGE;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { guardedEvaluate: guardedEvaluate, install: install };
    }
})(typeof window !== 'undefined' ? window : globalThis);
