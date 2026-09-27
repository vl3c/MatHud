// Output sanitiser for MathJax-typeset chat math (the backstop layer).
//
// Chat text comes from the model, the user and saved workspaces, so math in it is
// untrusted TeX. The MathJax configuration in templates/index.html is the first line
// of defence (no html extension, no \require, no user macros, a length cap, the
// ui/safe filter with URLs off, a fontfamily filter). This file is the second. It runs
// as a MathJax render action (registered in templates/index.html) right after each
// formula is inserted, on first render and on every re-render, including those the
// MathJax context menu triggers; ChatUIManager.render_math also runs it over the
// whole chat after each typeset. It walks the output of every mjx-container and
// removes anything math never needs:
//   - links: <a> elements are unwrapped; href, xlink:href and src attributes removed;
//   - event-handler attributes (on*);
//   - ids that do not start with "mjx-", and classes MathJax does not use itself;
//   - style declarations that could overlay the page or load or run anything:
//     position other than static/relative, large or non-length offsets (top, left,
//     right, bottom, inset), viewport units, z-index, cursor, content, and any value
//     with url(), image-set(), expression(), javascript: or @import.
// MathJax's own inline layout (position: relative with offsets of a fraction of an
// em, transforms, widths, margins) and colours, backgrounds without url(), borders,
// padding and font settings stay, so legitimate math looks the same. Each formula is
// also painted only inside its own box (the containment rules in static/style.css), so
// nested offsets such as \raise inside \raise cannot reach other messages.
//
// Timing: a url() in a style can be requested before this runs, so the MathJax
// configuration has to block network access on its own; this layer only guarantees
// the rendered result is inert (no links, overlays or foreign ids/classes).
(function () {
    "use strict";

    var BLOCKED_PROPERTY = /^(z-index|cursor|content|behavior|-moz-binding)$/i;
    var OFFSET_PROPERTY = /^(top|left|right|bottom|inset(-.*)?)$/i;
    // MathJax's own offsets are a fraction of an em or 0px; anything else is dropped.
    var SMALL_OFFSET = /^-?(\d*\.?\d+(em|ex)|\d{0,2}(\.\d+)?px|0)$/i;
    var ALLOWED_POSITION = /^(static|relative)$/i;
    var URL_CAPABLE_PROPERTY = /^(background|border-image|list-style|mask|-webkit-mask|filter|src)/i;
    var DANGEROUS_VALUE = /url\s*\(|image-set\s*\(|expression\s*\(|javascript\s*:|@import/i;
    // Viewport (vw, vh, vi, vb, vmin, vmax), small/large/dynamic viewport (sv*, lv*, dv*)
    // and container-query (cqw, cqh, cqi, cqb, cqmin, cqmax) units.
    var VIEWPORT_UNIT = /\d([sld]?v(w|h|i|b|min|max)|cq(w|h|i|b|min|max))\b/i;
    var ALLOWED_CLASS = /^(mjx-|MJX|MathJax|CtxtMenu_|TEX-)/;
    var ALLOWED_ID = /^mjx-/;
    var LINK_ATTRIBUTES = ["href", "xlink:href", "src", "action", "formaction"];
    var MAX_OFFSET_EM = 10;

    function isSmallOffset(value) {
        if (!SMALL_OFFSET.test(value)) {
            return false;
        }
        return !/(em|ex)$/i.test(value) || Math.abs(parseFloat(value)) <= MAX_OFFSET_EM;
    }

    function isBlockedDeclaration(name, value, positionAllowed) {
        if (BLOCKED_PROPERTY.test(name) || DANGEROUS_VALUE.test(value) || VIEWPORT_UNIT.test(value)) {
            return true;
        }
        if (name.toLowerCase() === "position") {
            return !positionAllowed;
        }
        if (OFFSET_PROPERTY.test(name)) {
            // Offsets of an element whose position was dropped go too.
            return !positionAllowed || !isSmallOffset(value.trim());
        }
        return URL_CAPABLE_PROPERTY.test(name) && /url\s*\(/i.test(value);
    }

    function sanitizeStyle(element) {
        var raw = element.getAttribute("style");
        if (raw === null) {
            return;
        }
        var style = element.style;
        var position = style.getPropertyValue("position").trim();
        var positionAllowed = !position || ALLOWED_POSITION.test(position);
        for (var i = style.length - 1; i >= 0; i--) {
            var name = style[i];
            if (isBlockedDeclaration(name, style.getPropertyValue(name), positionAllowed)) {
                style.removeProperty(name);
            }
        }
        // Re-serialise from the parsed declarations, dropping text the browser ignored.
        var cleaned = style.cssText;
        if (!cleaned || DANGEROUS_VALUE.test(cleaned)) {
            element.removeAttribute("style");
        } else if (cleaned !== raw) {
            element.setAttribute("style", cleaned);
        }
    }

    function sanitizeClasses(element) {
        if (!element.hasAttribute("class")) {
            return;
        }
        var kept = [];
        var names = element.getAttribute("class").split(/\s+/);
        for (var i = 0; i < names.length; i++) {
            if (names[i] && ALLOWED_CLASS.test(names[i])) {
                kept.push(names[i]);
            }
        }
        if (kept.length) {
            element.setAttribute("class", kept.join(" "));
        } else {
            element.removeAttribute("class");
        }
    }

    function sanitizeAttributes(element) {
        for (var i = 0; i < LINK_ATTRIBUTES.length; i++) {
            element.removeAttribute(LINK_ATTRIBUTES[i]);
        }
        var attributes = Array.prototype.slice.call(element.attributes);
        for (var j = 0; j < attributes.length; j++) {
            var name = attributes[j].name.toLowerCase();
            if (name.indexOf("on") === 0 || name.slice(-4) === "href") {
                element.removeAttribute(attributes[j].name);
            }
        }
        var id = element.getAttribute("id");
        if (id !== null && !ALLOWED_ID.test(id)) {
            element.removeAttribute("id");
        }
        sanitizeClasses(element);
        sanitizeStyle(element);
    }

    function unwrapLinks(container) {
        var links = container.querySelectorAll("a");
        for (var i = 0; i < links.length; i++) {
            var link = links[i];
            var parent = link.parentNode;
            if (!parent) {
                continue;
            }
            while (link.firstChild) {
                parent.insertBefore(link.firstChild, link);
            }
            parent.removeChild(link);
        }
    }

    function sanitizeContainer(container) {
        unwrapLinks(container);
        sanitizeAttributes(container);
        var elements = container.querySelectorAll("*");
        for (var i = 0; i < elements.length; i++) {
            sanitizeAttributes(elements[i]);
        }
    }

    // Set on each container this file has sanitised. TeX cannot put it on an
    // mjx-container (MathJax creates the container; ui/safe filters data- attributes).
    var SANITIZED_MARKER = "data-mathud-sanitized";

    // Sanitise every mjx-container inside root (or root itself when it is one). With
    // skipSanitized, containers already carrying the marker are left alone: the render
    // action has sanitised them, and sanitises them again whenever MathJax re-renders.
    // Returns the number of containers sanitised.
    function sanitize(root, skipSanitized) {
        if (!root || !root.querySelectorAll) {
            return 0;
        }
        var containers = Array.prototype.slice.call(root.querySelectorAll("mjx-container"));
        if (root.tagName && root.tagName.toLowerCase() === "mjx-container") {
            containers.unshift(root);
        }
        var count = 0;
        for (var i = 0; i < containers.length; i++) {
            if (skipSanitized && containers[i].hasAttribute(SANITIZED_MARKER)) {
                continue;
            }
            sanitizeContainer(containers[i]);
            containers[i].setAttribute(SANITIZED_MARKER, "");
            count++;
        }
        return count;
    }

    // Typeset root with MathJax, then sanitise any of its math the render action has
    // not (belt and braces). Resolves to the number of containers sanitised in this
    // pass; never rejects (a failed typeset still gets sanitised).
    //
    // Typesets run one after another, chained on MathJax.startup.promise as MathJax's
    // documentation recommends. typesetPromise is not queued: when a formula needs a TeX
    // extension that is not loaded yet (\ce, \bra, ...), MathJax loads it and then
    // renders again whatever the latest typeset asked for, so a typeset started during
    // the load left the pending formula as an undefined command. Even without a load,
    // the typeset runs after this returns: the math is in place once the promise resolves.
    function typesetAndSanitize(root) {
        var mathjax = window.MathJax;
        if (!mathjax || typeof mathjax.typesetPromise !== "function") {
            return Promise.resolve(sanitize(root, true));
        }
        function sanitizeRoot() {
            return sanitize(root, true);
        }
        function typeset() {
            try {
                return mathjax.typesetPromise([root]).then(sanitizeRoot, sanitizeRoot);
            } catch (error) {
                return sanitizeRoot();
            }
        }
        var startup = mathjax.startup;
        var previous = startup && startup.promise ? startup.promise : Promise.resolve();
        var done = previous.then(typeset, typeset);
        if (startup) {
            startup.promise = done;
        }
        return done;
    }

    window.MatHudMathSafety = {
        sanitize: sanitize,
        typesetAndSanitize: typesetAndSanitize,
    };
})();
