"""
Utility module for JSON-related operations in the photometry service.
"""

import json
import os
from dataclasses import asdict
from typing import Optional

from forcedphot.image_photometry.utils import EndResult


def save_results_to_json(end_result: EndResult, output_dir: str, filename: Optional[str] = None) -> str:
    """
    Save photometry end results to a JSON file.

    Parameters
    ----------
    end_result : EndResult
        The end results to save
    output_dir : str
        Directory where to save the JSON file
    filename : str, optional
        Custom filename for the JSON file. If not provided, will generate one
        based on target name and visit ID

    Returns
    -------
    str
        Path to the saved JSON file

    Raises
    ------
    OSError
        If there's an error creating the output directory or writing the file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        sanitized_name = end_result.target_name.replace(" ", "_").lower()
        filename = f"{sanitized_name}_visit{end_result.visit_id}.json"

    # Ensure filename has .json extension
    if not filename.endswith('.json'):
        filename += '.json'

    output_path = os.path.join(output_dir, filename)

    try:
        # Convert EndResult to dictionary
        result_dict = asdict(end_result)

        # Save to JSON file with proper formatting
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")
        return output_path

    except Exception as e:
        raise OSError(f"Failed to save results to JSON: {str(e)}") from e
