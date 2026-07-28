# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Flask API for Information Quality Metrics."""

import importlib
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
from flasgger import Swagger
from flask import Flask, jsonify, render_template, request

from infqm.base_metric import BaseMetric

# Add metrics folder to path
METRICS_PATH = "src/kpi_num"
sys.path.insert(0, METRICS_PATH)

app = Flask(__name__)

template = {
    "swagger": "2.0",
    "info": {
        "title": "Information Measurement API",
        "description": "This API is used to measure the information quality within "
        "given images. It is part of the TP3 within the project SAIFE - "
        "Safe AI Engineering",
        "version": "1.0",
    },
}

app.config["SWAGGER"] = {
    "title": "Flask Kafka API",
    "uiversion": 2,
    "template": "./resources/flasgger/swagger_ui.html",
}

Swagger(app, template=template)


@app.route("/")
def index():
    """Home page.

    Returns:
        Rendered HTML template for the home page.
    """
    return render_template("index.html")


# Store loaded metrics
loaded_metrics = {}


def discover_and_load_metrics() -> None:
    """Dynamically discover and load all metric classes."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.parent
    metrics_dir = script_dir / "kpi_num"

    # Add metrics directory to Python path
    if str(metrics_dir) not in sys.path:
        sys.path.insert(0, str(metrics_dir))

    # Find all Python files in the metrics directory
    metric_files = []
    if metrics_dir.exists():
        metric_files.extend(
            file_path.stem
            for file_path in metrics_dir.glob("*.py")
            if file_path.name != "__init__.py"
        )

    # Load each metric module and instantiate metric classes
    for module_name in metric_files:
        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Find all classes in the module that inherit from BaseMetric
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseMetric)
                    and obj != BaseMetric
                ):
                    # Instantiate the metric class
                    metric_instance_member = obj()

                    name.replace("Metric", "").lower()
                    loaded_metrics[metric_instance_member.get_name()] = (
                        metric_instance_member
                    )

        except (ImportError, AttributeError, TypeError):
            pass


def load_image_from_request():
    """Load an image from the incoming request.

    Load image from request (either file upload or base64) Returns
    OpenCV image array.

    Returns:
        Tuple of (cv_image, error_message). cv_image is the
        loaded image or None if there was an error.
    """
    if "image" not in request.files:
        return None, "No image file provided"

    file = request.files["image"]
    if file.filename == "":
        return None, "No image file selected"

    try:
        # Read image data
        file_bytes = np.frombuffer(file.read(), np.uint8)
        cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if cv_image is None:
            return None, "Invalid image format"

        return cv_image, None

    except (ValueError, RuntimeError, TypeError) as e:
        return None, f"Error processing image: {e!s}"


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint.

    Returns:
        JSON response indicating the health status and number of
        loaded metrics.
    """
    return jsonify({"status": "healthy", "loaded_metrics": len(loaded_metrics)})


def create_metric_endpoint(metric_name, metric_instance):
    """Create a dynamic endpoint for a specific metric.

    Args:
        metric_name: Name of the metric (used in the endpoint URL)
        metric_instance: An instance of the metric class to use for calculations

    Returns:
        A Flask view function that calculates the metric for a
        given image.
    """

    def metric_endpoint():
        """Endpoint to calculate a specific metric for a given image.

        Returns:
            JSON response containing the metric result or an error message.
        """
        if request.method != "POST":
            return jsonify({"error": "Only POST method is allowed"}), 405

        # Load image from request
        cv_image, error = load_image_from_request()
        if error:
            return jsonify({"error": error}), 400

        try:
            # Calculate metric
            result = metric_instance.calculate(cv_image)

            return jsonify({
                "metric": metric_name,
                "value": float(result),
                "image_shape": cv_image.shape,
                "status": "success",
            })

        except (ValueError, RuntimeError, TypeError) as e:
            return (
                jsonify({
                    "metric": metric_name,
                    "error": f"Error calculating metric: {e!s}",
                    "status": "error",
                }),
                500,
            )

    return metric_endpoint


@app.route("/batch", methods=["POST"])
def batch_metrics():
    """Calculate all metrics for a single image.

    Returns:
        JSON response containing the results of all
        metrics and any errors encountered during calculation.
    """
    # Load image from request
    cv_image, error = load_image_from_request()
    if error:
        return jsonify({"error": error}), 400

    results = {}
    errors = {}

    for metric_name, metric_instance in loaded_metrics.items():
        try:
            result = metric_instance.calculate(cv_image)
            results[metric_name] = float(result)
        except (ValueError, RuntimeError, TypeError) as e:
            errors[metric_name] = str(e)

    response = {
        "results": results,
        "image_shape": cv_image.shape,
        "status": "success" if not errors else "partial_success",
    }

    if errors:
        response["errors"] = errors

    return jsonify(response)


# Initialize the application
if __name__ == "__main__":
    # Discover and load all metrics
    discover_and_load_metrics()

    # Create dynamic endpoints for each metric
    for metric_name_str, metric_instance_module in loaded_metrics.items():
        endpoint_func = create_metric_endpoint(metric_name_str, metric_instance_module)
        endpoint_func.__name__ = f"{metric_name_str}_endpoint"  # Give unique name

        endpoint_func.__doc__ = f"""
        {metric_name_str} endpoint
        ---
        post:
            description: Run metric '{metric_name_str}' using the provided payload
        tags:
            - Numerical KPIs
        parameters:
            -   name: image
                in: formData
                type: file
                required: true
                description: The image file to evaluate
        responses:
            200:
                description: Result of {metric_name_str}
        """

        # Add the route
        app.add_url_rule(
            f"/{metric_name_str}",
            endpoint=metric_name_str,
            view_func=endpoint_func,
            methods=["POST"],
        )

    app.run(debug=True)
