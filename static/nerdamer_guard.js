/*
 * Time budget for nerdamer computations.
 *
 * nerdamer 1.1.13 applies L'Hopital's rule in an unbounded loop (Calculus.js, Limit.divide), so a
 * limit such as limit(abs(x)/x, x, Infinity) never returns and freezes the browser tab: the
 * derivative of abs(x) is abs(x)/x, which gives back the same quotient on every pass. Other
 * computations can run for minutes too, e.g. expand((x+y+z)^25).
 *
 * window.MatHudGuardedNerdamer(expression, maxSteps, maxMs, decimals) evaluates a nerdamer
 * expression (evaluated and printed as decimals when decimals is true) under a budget:
 * - every derivative and recursive limit step counts against maxSteps (default 2000);
 * - the parser's arithmetic (add, subtract, multiply, divide, pow, expand, parse), simplify,
 *   factor, the algebraic division and integration check the clock every 32 calls, so a
 *   computation stops within milliseconds of the deadline (maxMs, default 1500) whichever
 *   step it is in.
 * When the budget runs out the next checked call throws; once it has thrown, every checked call
 * throws, so nerdamer's own try/catch blocks cannot resume the computation. nerdamer's settings
 * (which its block() helper changes without restoring them on an exception) are restored
 * afterwards. The result is {text, exceeded, error, guarded, elapsedMs}, with error '' when
 * nerdamer raised nothing. Outside a guarded call every wrapper is a single flag test.
 */
(function (root) {
    'use strict';

    var DEFAULT_MAX_STEPS = 2000;
    var DEFAULT_MAX_MS = 1500;
    var CLOCK_EVERY = 32;
    var BUDGET_MESSAGE = 'MatHud: symbolic computation budget exceeded';

    var budget = { active: false, steps: 0, ticks: 0, maxSteps: 0, deadline: 0, exceeded: false };

    function trip() {
        budget.exceeded = true;
        throw new Error(BUDGET_MESSAGE);
    }

    // A counted step: derivatives and limit steps.
    function spend() {
        if (!budget.active) {
            return;
        }
        budget.steps += 1;
        if (budget.exceeded || budget.steps > budget.maxSteps || Date.now() > budget.deadline) {
            trip();
        }
    }

    // A clock check for frequent calls: the time is read every CLOCK_EVERY calls.
    function tick() {
        if (!budget.active) {
            return;
        }
        if (budget.exceeded) {
            trip();
        }
        budget.ticks += 1;
        if (budget.ticks % CLOCK_EVERY === 0 && Date.now() > budget.deadline) {
            trip();
        }
    }

    function wrap(owner, key, check) {
        var original = owner && owner[key];
        if (typeof original !== 'function' || original.__mathudGuarded) {
            return false;
        }
        var guarded = function () {
            check();
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
        var core = nerdamer.getCore();
        var calculus = core.Calculus;
        if (!calculus || !calculus.Limit) {
            return false;
        }
        // These are looked up as properties on every call, so the wrappers see recursive calls.
        wrap(calculus, 'diff', spend);
        wrap(calculus.Limit, 'limit', spend);
        wrap(calculus.Limit, 'divide', spend);
        wrap(calculus, 'integrate', tick);
        var parser = core.PARSER;
        ['add', 'subtract', 'multiply', 'divide', 'pow', 'expand', 'parse'].forEach(function (key) {
            wrap(parser, key, tick);
        });
        // Printing and hashing large symbols (text(), which simplify uses to compare terms)
        // runs for seconds without touching the parser.
        var symbolMethods = core.Symbol && core.Symbol.prototype;
        ['collectSymbols', 'updateHash', 'insert', 'text'].forEach(function (key) {
            wrap(symbolMethods, key, tick);
        });
        var algebra = core.Algebra;
        if (algebra) {
            wrap(algebra, 'divide', tick);
            wrap(algebra.Simplify, 'simplify', tick);
            wrap(algebra.Factor, 'factor', tick);
        }
        return true;
    }

    function snapshot(settings) {
        var copy = {};
        Object.keys(settings).forEach(function (key) {
            copy[key] = settings[key];
        });
        return copy;
    }

    function restore(settings, saved) {
        Object.keys(settings).forEach(function (key) {
            if (!Object.prototype.hasOwnProperty.call(saved, key)) {
                delete settings[key];
            }
        });
        Object.keys(saved).forEach(function (key) {
            settings[key] = saved[key];
        });
    }

    function compute(expression, decimals) {
        var parsed = root.nerdamer(expression);
        return String(decimals ? parsed.evaluate().text('decimals') : parsed.text());
    }

    function guardedEvaluate(expression, maxSteps, maxMs, decimals) {
        var installed = install(root.nerdamer);
        // Empty strings rather than null: Brython does not map a JavaScript null to None.
        var result = { text: '', exceeded: false, error: '', guarded: installed, elapsedMs: 0 };
        if (budget.active) {
            // Nested call: share the outer budget.
            result.text = compute(expression, decimals);
            return result;
        }
        var settings = installed ? root.nerdamer.getCore().Settings : null;
        var saved = settings ? snapshot(settings) : null;
        var started = Date.now();
        budget.active = true;
        budget.steps = 0;
        budget.ticks = 0;
        budget.maxSteps = maxSteps || DEFAULT_MAX_STEPS;
        budget.deadline = started + (maxMs || DEFAULT_MAX_MS);
        budget.exceeded = false;
        try {
            result.text = compute(expression, decimals);
        } catch (error) {
            result.error = String((error && error.message) || error);
        } finally {
            result.exceeded = budget.exceeded;
            budget.active = false;
            if (saved && (result.exceeded || result.error)) {
                restore(settings, saved);
            }
            result.elapsedMs = Date.now() - started;
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
