from fastapi import FastAPI, Query, HTTPException
from app.alma_utils import (
    query_by_targets,
    query_by_science,
    fetch_science_types,
)
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ALMASim API",
              description="Advanced querying for ALMA data.")


@app.get("/science_keywords/")
def get_science_keywords():
    """
    Fetch science keywords and categories from ALMA TAP.
    """
    try:
        logger.info("Fetching science keywords and categories...")
        keywords, categories = fetch_science_types()
        logger.info("Fetched science keywords and categories successfully.")
        return {"science_keywords": keywords,
                "scientific_categories": categories}
    except Exception as e:
        logger.error(f"Error fetching science keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_targets/")
def query_targets(targets: list[dict]):
    """
    Query ALMA observations based on a target list.

    Example `targets`:
    [
        {"target_name": "NGC253", "member_ous_uid": "uid://A001/X122/X1"},
        {"target_name": "M87", "member_ous_uid": "uid://A001/X456/X2"}
    ]
    """
    try:
        logger.info(f"Received target query request: {targets}")
        # Validate target structure
        for target in targets:
            if "target_name" not in target or "member_ous_uid" not in target:
                raise ValueError(
                    "Each target must include 'target_name' and \
                    'member_ous_uid'."
                )

        target_list = [
            (target["target_name"],
             target["member_ous_uid"]) for target in targets
        ]
        results = query_by_targets(target_list)
        if results.empty:
            logger.info("No observations found for the provided targets.")
            return {"message": "No observations found."}
        logger.info("Query successful, returning results.")
        return results.to_dict(orient="records")
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error querying targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_science/")
def query_science(
    science_keywords: list[str] = Query(None),
    scientific_categories: list[str] = Query(None),
    bands: list[int] = Query(None),
    frequency_min: float = Query(None),
    frequency_max: float = Query(None),
    spatial_res_min: float = Query(None),
    spatial_res_max: float = Query(None),
    velocity_res_min: float = Query(None),
    velocity_res_max: float = Query(None),
):
    """
    Query ALMA observations based on science-related filters.
    """
    try:
        logger.info("Received science query request.")
        science_filters = {
            "science_keywords": science_keywords,
            "scientific_categories": scientific_categories,
            "bands": bands,
            "frequency": (
                (frequency_min, frequency_max)
                if frequency_min and frequency_max
                else None
            ),
            "spatial_resolution": (
                (spatial_res_min, spatial_res_max)
                if spatial_res_min and spatial_res_max
                else None
            ),
            "velocity_resolution": (
                (velocity_res_min, velocity_res_max)
                if velocity_res_min and velocity_res_max
                else None
            ),
        }
        science_filters = {
            k: v for k, v in science_filters.items() if v is not None
        }
        logger.info(f"Science filters: {science_filters}")

        results = query_by_science(science_filters)
        if results.empty:
            logger.info("No observations found for the provided science \
                        filters.")
            return {"message": "No observations found."}
        logger.info("Query successful, returning results.")
        return results.to_dict(orient="records")
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error querying science filters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)