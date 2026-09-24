// Tight path-tracing loops for the Canvas2D renderer.
//
// Brython makes per-point Python work expensive, so Canvas2DPrimitiveAdapter hands
// whole batches of paths to these helpers instead of issuing one moveTo/lineTo per
// point from Python. Each path is an array of [x, y] pairs (Brython converts Python
// tuples/lists automatically). The adapter keeps a pure-Python fallback when these
// helpers are unavailable.
(function () {
    "use strict";

    function tracePath(ctx, path) {
        var n = path.length;
        ctx.moveTo(path[0][0], path[0][1]);
        for (var j = 1; j < n; j++) {
            ctx.lineTo(path[j][0], path[j][1]);
        }
    }

    // Stroke every open path as its own subpath with a single stroke() call.
    window.MatHudStrokePaths = function (ctx, paths) {
        ctx.beginPath();
        for (var i = 0; i < paths.length; i++) {
            if (paths[i].length >= 2) {
                tracePath(ctx, paths[i]);
            }
        }
        ctx.stroke();
    };

    // Close and fill (fill=true) or stroke (fill=false) each polygon separately so
    // overlapping translucent polygons keep compositing individually.
    window.MatHudTracePolygons = function (ctx, polygons, fill) {
        var drawn = 0;
        for (var i = 0; i < polygons.length; i++) {
            if (polygons[i].length < 3) {
                continue;
            }
            ctx.beginPath();
            tracePath(ctx, polygons[i]);
            ctx.closePath();
            if (fill) {
                ctx.fill();
            } else {
                ctx.stroke();
            }
            drawn++;
        }
        return drawn;
    };
})();
