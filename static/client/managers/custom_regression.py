"""Custom-model fits for ``fit_regression``: checks against the canvas, the fit, and its parameters.

``fit_regression`` with model_type "custom" fits a model with named constants
(``a*exp(-b*x)*cos(c*x) + d``) by least squares (utils/statistics/custom_regression.py)
and turns the fitted constants into parameters (sliders), so the plotted curve, whose
expression is the model itself, follows them.

``CustomRegression.prepare`` reads the model, checks it against the canvas and fits it
without changing anything; ``apply`` then sets the parameters. StatisticsManager draws the
curve and the data points around these two calls, inside one undo step.

Rules:
    - A constant may not take the name of another object on the canvas (point A).
    - A constant naming an existing parameter is fitted from its value and updated. When
      other objects read that parameter, the call is refused unless update_parameters is
      true, and then the result lists them in moved_objects. A function whose expression
      is the model itself does not count: it is the curve being fitted, and it is not
      drawn again.

Pure Python (no browser imports), so server-side pytest imports it directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from drawables.parameter import range_around
from expression_validator import ExpressionValidator
from utils.statistics.regression import residual_report

if TYPE_CHECKING:
    from managers.drawables_container import DrawablesContainer
    from managers.parameter_manager import ParameterManager
    from utils.statistics.custom_model import CustomModel
    from utils.statistics.custom_regression import CustomFit

CUSTOM_MODEL_TYPE = "custom"
# Significant digits of the constants substituted into the reported expression
EXPRESSION_DIGITS = 6


class CustomFitPlan:
    """A checked custom fit, ready to apply.

    Attributes:
        existing: Constants that name existing parameters (name -> parameter).
        own_function: An existing function whose expression is the model, or None.
        moved: Names of other objects that read the existing parameters (update_parameters).
    """

    __slots__ = ("model", "fit", "y_data", "existing", "own_function", "moved")

    def __init__(
        self,
        model: CustomModel,
        fit: CustomFit,
        y_data: List[float],
        existing: Dict[str, Any],
        own_function: Optional[Any],
        moved: List[str],
    ) -> None:
        self.model = model
        self.fit = fit
        self.y_data = y_data
        self.existing = existing
        self.own_function = own_function
        self.moved = moved


class CustomRegression:
    """Prepares and applies custom-model fits on one canvas."""

    def __init__(self, drawables: "DrawablesContainer", parameters: Optional["ParameterManager"]) -> None:
        self.drawables = drawables
        self.parameters = parameters

    def prepare(
        self,
        model_text: Any,
        x_data: Sequence[float],
        y_data: Sequence[float],
        initial_values: Any,
        update_parameters: Optional[bool],
    ) -> CustomFitPlan:
        """Read, check and fit the model; nothing on the canvas changes.

        Raises:
            ValueError: With a hint, for a bad model, constant name, initial value or data,
                or existing parameters other objects read (without update_parameters).
        """
        # Imported on first use: the fitting engine is only compiled (by Brython) when needed
        from utils.statistics.custom_model import read_model
        from utils.statistics.custom_regression import fit_custom_model

        if self.parameters is None:
            raise ValueError("Custom fits need the canvas's parameters, which are not available here.")
        model = read_model(model_text)
        existing = self._existing_parameters(model)
        self._check_free_names(model, existing)
        own_function = self._function_drawing(model)
        moved = self._moved_objects(existing, own_function, update_parameters)
        starts = starting_values(model, initial_values, existing)
        fit = fit_custom_model(x_data, y_data, model, starts)
        return CustomFitPlan(model, fit, [float(y) for y in y_data], existing, own_function, moved)

    def apply(self, plan: CustomFitPlan) -> List[str]:
        """Set each constant's parameter to its fitted value; return notes on widened ranges.

        A new parameter gets a slider range around its value (drawables.parameter.range_around);
        an existing one keeps its range unless the value falls outside it.
        """
        assert self.parameters is not None
        notes: List[str] = []
        for name in plan.model.constants:
            value = plan.fit.values[name]
            if name in plan.existing:
                report = self.parameters.set_parameter(name, value)
            else:
                low, high, step = range_around(value)
                report = self.parameters.set_parameter(name, value, low, high, step)
            notes.extend(f"Parameter {name}: {note}." for note in report.get("notes", []))
        return notes

    @staticmethod
    def result_fields(plan: CustomFitPlan, parameter_notes: List[str]) -> Dict[str, Any]:
        """The custom-specific keys of fit_regression's result."""
        model, fit = plan.model, plan.fit
        fields: Dict[str, Any] = {
            "model_type": CUSTOM_MODEL_TYPE,
            "model": model.text,
            "expression": model.with_values(fit.values, EXPRESSION_DIGITS),
            "coefficients": dict(fit.values),
            "parameters": list(model.constants),
            "created_parameters": [name for name in model.constants if name not in plan.existing],
            "r_squared": fit.r_squared,
        }
        fields.update(residual_report(plan.y_data, fit.predictions))
        notes = list(fit.notes) + parameter_notes
        if plan.own_function is not None:
            notes.append(
                f"Function {plan.own_function.name} already draws this model, so no new curve was drawn; "
                f"it now shows the fit."
            )
        if plan.moved:
            fields["moved_objects"] = list(plan.moved)
            notes.append(f"{', '.join(plan.moved)} read the updated parameters and moved with them.")
        if notes:
            fields["notes"] = notes
        return fields

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _existing_parameters(self, model: CustomModel) -> Dict[str, Any]:
        assert self.parameters is not None
        existing: Dict[str, Any] = {}
        for name in model.constants:
            parameter = self.parameters.get_parameter(name)
            if parameter is not None:
                existing[name] = parameter
        return existing

    def _check_free_names(self, model: CustomModel, existing: Dict[str, Any]) -> None:
        """Refuse a constant named like another object on the canvas (a point A)."""
        in_use = self.drawables.names_in_use()
        for name in model.constants:
            if name not in existing and name in in_use:
                raise ValueError(
                    f"The model's constant '{name}' is the name of another object on the canvas; "
                    f"rename the constant in the model, such as {name}1."
                )

    def _function_drawing(self, model: CustomModel) -> Optional[Any]:
        """An existing function whose expression is the model, or None."""
        wanted = _compact(ExpressionValidator.fix_math_expression(model.text))
        for function in self.drawables.get_by_class_name("Function"):
            if _compact(str(getattr(function, "function_string", ""))) == wanted:
                return function
        return None

    def _moved_objects(
        self, existing: Dict[str, Any], own_function: Optional[Any], update_parameters: Optional[bool]
    ) -> List[str]:
        """Other objects that read the existing parameters; refused unless update_parameters."""
        if not existing:
            return []
        assert self.parameters is not None
        users = [user for user in self.parameters.users_of(list(existing)) if user is not own_function]
        if not users:
            return []
        names = [str(user.name) for user in users]
        if update_parameters is not True:
            constants = ", ".join(existing)
            renamed = ", ".join(f"{name}2" for name in existing)
            raise ValueError(
                f"Parameters {constants} already exist and {', '.join(names)} read them, so fitting would move "
                f"those objects. Name the model's constants differently (such as {renamed}), or pass "
                f"update_parameters: true to fit them and move those objects."
            )
        return names


def starting_values(model: CustomModel, initial_values: Any, existing: Dict[str, Any]) -> Dict[str, float]:
    """Starting guesses: existing parameters' values, then initial_values ({name, value} items).

    Raises:
        ValueError: initial_values is not a list of {name, value}, names something other
            than a constant of the model (twice), or gives a value that is not a finite number.
    """
    starts = {name: float(parameter.value) for name, parameter in existing.items()}
    if initial_values is None:
        return starts
    constants = ", ".join(model.constants)
    if not isinstance(initial_values, list):
        raise ValueError("initial_values must be a list of {name, value} items, such as [{name: 'b', value: 2}].")
    given = set()
    for item in initial_values:
        if not isinstance(item, dict) or "name" not in item or "value" not in item:
            raise ValueError(f"Each initial_values item must be {{name, value}}, got {item!r}.")
        name = str(item["name"]).strip()
        if name not in model.constants:
            raise ValueError(
                f"initial_values names '{name}', which is not a constant of the model; its constants are {constants}."
            )
        if name in given:
            raise ValueError(f"initial_values gives '{name}' twice; give each constant once.")
        given.add(name)
        starts[name] = _finite(item["value"], f"The initial value of {name}")
    return starts


def refuse_custom_arguments(model: Any, initial_values: Any, update_parameters: Any) -> None:
    """model, initial_values and update_parameters belong to custom fits only."""
    for label, value in (("model", model), ("initial_values", initial_values)):
        if value is not None:
            raise ValueError(f"{label} is only read with model_type '{CUSTOM_MODEL_TYPE}'; pass null.")
    if update_parameters:
        raise ValueError(f"update_parameters is only read with model_type '{CUSTOM_MODEL_TYPE}'; pass null.")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {value!r}.")
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    return number


def _compact(text: str) -> str:
    return "".join(text.split())
