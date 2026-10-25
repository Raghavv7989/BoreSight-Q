"""
BoreSight-Q: Hydrogeological Schemas and Data Models.

Defines the core data contracts for parcel requests, prioritized borewell points,
geospatial feature collections, and hydrogeological analysis responses.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class GeoJSONPolygon(BaseModel):
    """GeoJSON Polygon Geometry definition in EPSG:4326."""
    type: Literal["Polygon"] = Field(
        default="Polygon",
        description="Must be GeoJSON geometry type 'Polygon'."
    )
    coordinates: List[List[List[float]]] = Field(
        ...,
        description="Linear rings coordinates: [[[lon, lat], [lon, lat], ...]]. First and last point must match."
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: List[List[List[float]]]) -> List[List[List[float]]]:
        if not v or len(v) < 1:
            raise ValueError("Polygon coordinates must contain at least one linear ring.")
        exterior_ring = v[0]
        if len(exterior_ring) < 4:
            raise ValueError("Exterior linear ring must have at least 4 coordinates (closed ring).")
        for pt in exterior_ring:
            if len(pt) < 2:
                raise ValueError("Coordinate points must have at least [lon, lat].")
            lon, lat = pt[0], pt[1]
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"Longitude {lon} is out of bounds [-180, 180].")
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"Latitude {lat} is out of bounds [-90, 90].")
        return v


class ParcelRequest(BaseModel):
    """
    Parcel-scale analysis request payload.
    Takes a parcel boundary polygon in WGS84 (EPSG:4326) and a hydrogeological buffer in meters.
    """
    parcel_geometry: Union[GeoJSONPolygon, Dict[str, Any]] = Field(
        ...,
        description="GeoJSON Polygon geometry representing the land parcel in EPSG:4326 coordinates."
    )
    buffer_m: float = Field(
        default=1500.0,
        ge=100.0,
        le=10000.0,
        description="Buffer radius around parcel in meters for regional hydrogeological and fracture analysis."
    )

    @field_validator("parcel_geometry")
    @classmethod
    def validate_geometry_dict(cls, v: Union[GeoJSONPolygon, Dict[str, Any]]) -> Union[GeoJSONPolygon, Dict[str, Any]]:
        if isinstance(v, dict):
            geom_type = v.get("type")
            if geom_type != "Polygon":
                raise ValueError(f"Expected GeoJSON geometry type 'Polygon', got '{geom_type}'")
            coords = v.get("coordinates")
            if not coords or not isinstance(coords, list):
                raise ValueError("Polygon geometry must contain a 'coordinates' list.")
        return v


class BorewellPoint(BaseModel):
    """
    Recommended borewell drilling target location (P1 primary, P2 secondary, P3 contingency)
    derived from multi-criteria hydrogeological, structural lineament, and SAR velocity analyses.
    """
    id: Literal["P1", "P2", "P3"] = Field(
        ...,
        description="Point identifier: 'P1' (Optimal), 'P2' (Secondary), 'P3' (Contingency)."
    )
    lat: float = Field(..., description="WGS84 Latitude of recommended drilling point.")
    lon: float = Field(..., description="WGS84 Longitude of recommended drilling point.")
    gwpi_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Groundwater Potential Index score (0 to 100)."
    )
    confidence_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Exploration confidence percentage based on sensor convergence."
    )
    recommended_depth_m: float = Field(
        ...,
        ge=0.0,
        le=600.0,
        description="Recommended target drilling depth in meters below ground level (0m if unfeasible/barren)."
    )
    estimated_yield_lph: float = Field(
        ...,
        ge=0.0,
        description="Estimated sustainable borehole discharge yield in Litres Per Hour (LPH)."
    )
    delivery_pipe_size_inches: float = Field(
        default=1.5,
        ge=0.0,
        description="Recommended borehole delivery / column pipe diameter in inches."
    )
    pipe_size_display: str = Field(
        default='1.5 inches (1½")',
        description="Formatted display string for delivery pipe size."
    )
    recommended_pump_hp: str = Field(
        default="3.0 HP to 5.0 HP Submersible",
        description="Recommended submersible pump horsepower rating and type."
    )
    fracture_azimuth_deg: float = Field(
        ...,
        ge=0.0,
        le=360.0,
        description="Primary strike azimuth of controlling structural fracture or lineament in degrees (0-360°)."
    )
    stream_order: int = Field(
        ...,
        ge=0,
        le=10,
        description="Strahler drainage stream order intersecting or feeding the recharge zone."
    )
    reasoning: str = Field(
        ...,
        description="Detailed geological, structural, and hydrological explanation justifying point selection."
    )


class SoilGridsData(BaseModel):
    """Soil physical properties from ISRIC SoilGrids and derived hydraulic conductivity."""
    clay_g_kg: float = Field(..., description="Clay content in g/kg.")
    sand_g_kg: float = Field(..., description="Sand content in g/kg.")
    silt_g_kg: float = Field(..., description="Silt content in g/kg.")
    bulk_density_cg_cm3: float = Field(..., description="Bulk density in cg/cm3.")
    clay_pct: float = Field(..., description="Clay percentage (%).")
    sand_pct: float = Field(..., description="Sand percentage (%).")
    silt_pct: float = Field(..., description="Silt percentage (%).")
    bulk_density_g_cm3: float = Field(..., description="Bulk density in g/cm3.")
    ksat_mm_hr: float = Field(..., description="Saturated hydraulic conductivity (Ksat) in mm/hr.")
    ksat_m_day: float = Field(..., description="Saturated hydraulic conductivity (Ksat) in m/day.")
    soil_texture_class: str = Field(..., description="USDA soil texture classification.")


class InSarVelocitySummary(BaseModel):
    """Multi-temporal InSAR displacement and moisture trend proxy metrics."""
    mean_velocity_mm_yr: float = Field(..., description="Mean vertical deformation/subsidence velocity (mm/year).")
    min_velocity_mm_yr: float = Field(..., description="Minimum deformation velocity (maximum subsidence rate, mm/year).")
    max_velocity_mm_yr: float = Field(..., description="Maximum deformation velocity (maximum uplift/swelling rate, mm/year).")
    velocity_std_mm_yr: float = Field(..., description="Standard deviation of deformation velocity across parcel area.")
    temporal_baseline_months: int = Field(..., description="Duration of Sentinel-1 observation baseline window in months.")
    acquisitions_count: int = Field(..., description="Number of valid multi-temporal Sentinel-1 scenes processed.")
    compaction_trend: str = Field(..., description="Compaction/subsidence trend summary: 'Stable', 'Subsidence', 'Recharge Swelling'.")


class AquiferSummary(BaseModel):
    """Synthesized hydrogeological assessment of the aquifer system."""
    dominant_lithology: str = Field(..., description="Dominant geological bedrock formation.")
    aquifer_type: str = Field(..., description="Unconfined, Semi-confined, or Fractured Hard-rock.")
    soil_properties: SoilGridsData = Field(..., description="ISRIC SoilGrids profile and Ksat.")
    insar_dynamics: InSarVelocitySummary = Field(..., description="InSAR deformation and moisture dynamics.")
    recharge_potential: str = Field(..., description="Recharge rating: 'High', 'Moderate', or 'Low'.")
    permeability_rating: str = Field(..., description="Aquifer matrix permeability classification.")
    drilling_recommendation: str = Field(..., description="General drilling strategy and rotary/DTH rig guidance.")


class ParcelMetrics(BaseModel):
    """Accurate WGS84 geodesic land area and perimeter measurements."""
    area_sq_m: float = Field(..., description="Geodesic area in square meters.")
    area_acres: float = Field(..., description="Land area in acres.")
    acres_cents_str: str = Field(..., description="Formatted acres and cents string, e.g. '3 Ac 45 Cents'.")
    area_hectares: float = Field(..., description="Land area in hectares.")
    area_sq_yards: float = Field(..., description="Land area in square yards (gajam).")
    perimeter_m: float = Field(..., description="Total perimeter in meters.")
    perimeter_ft: float = Field(..., description="Total perimeter in feet.")


class HydroAnalysisResponse(BaseModel):
    """
    Complete parcel-scale hydrogeological analysis response payload.
    Contains prioritized drilling targets, structural lineaments, drainage networks,
    raster heatmaps encoded in base64, geographic bounds, and comprehensive aquifer summary.
    """
    borewell_points: List[BorewellPoint] = Field(
        ...,
        description="Top 3 ranked drilling points (P1, P2, P3) within or adjacent to the parcel."
    )
    drainage_geojson: Dict[str, Any] = Field(
        ...,
        description="GeoJSON FeatureCollection representing extracted drainage flowlines with Strahler orders."
    )
    lineaments_geojson: Dict[str, Any] = Field(
        ...,
        description="GeoJSON FeatureCollection representing structural fracture lineaments and fault lines."
    )
    intersections_geojson: Dict[str, Any] = Field(
        ...,
        description="GeoJSON FeatureCollection of structural-drainage and fracture-fracture intersection nodes."
    )
    gwpi_heatmap_base64: str = Field(
        ...,
        description="Base64 encoded PNG image (or data URI) of the Groundwater Potential Index (GWPI) heatmap."
    )
    insar_heatmap_base64: str = Field(
        ...,
        description="Base64 encoded PNG image (or data URI) of the multi-temporal InSAR deformation velocity (mm/yr) map."
    )
    sar_moisture_base64: str = Field(
        ...,
        description="Base64 encoded PNG image (or data URI) of the SAR surface moisture proxy distribution."
    )
    bounds: List[float] = Field(
        ...,
        description="Geographic bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326."
    )
    aquifer_summary: Union[AquiferSummary, Dict[str, Any]] = Field(
        ...,
        description="Hydrogeological synthesis including soil properties, hydraulic conductivity, and aquifer mechanics."
    )
    parcel_metrics: Optional[ParcelMetrics] = Field(
        default=None,
        description="WGS84 ellipsoidal geodesic land area and perimeter metrics."
    )
    is_hyper_arid: bool = Field(
        default=False,
        description="True if parcel lies in a hyper-arid desert zone (<50mm annual precipitation, NDVI < 0.05)."
    )
    aridity_warning: Optional[str] = Field(
        default=None,
        description="Warning advisory for hyper-arid desert regions."
    )
    is_surface_water: bool = Field(
        default=False,
        description="True if parcel is open surface water / ocean / river channel."
    )
    surface_water_warning: Optional[str] = Field(
        default=None,
        description="Warning advisory for open surface water / ocean / river channel regions."
    )

