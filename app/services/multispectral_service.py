import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import tempfile
import base64
import io
import logging
from typing import Dict, Any, List, Tuple
import asyncio
import re

logger = logging.getLogger(__name__)

class MultispectralAnalyzer:
    def __init__(self):
        self.crop_params = {
            'Cashew': {
                'temp_range': (24, 28),
                'moisture_range': (0.3, 0.5),
                'vegetation_range': (0.2, 0.4),
                'soil_range': (0.5, 0.7),
                'preferred_soil': [1, 3],  # Loamy, Organic
                'color': '#FFA500'  # Orange
            },
            'Cassava': {
                'temp_range': (25, 29),
                'moisture_range': (0.4, 0.6),
                'vegetation_range': (0.1, 0.3),
                'soil_range': (0.3, 0.5),
                'preferred_soil': [0, 1],  # Sandy, Loamy
                'color': '#8B4513'  # Brown
            },
            'Tomatoes': {
                'temp_range': (18, 24),
                'moisture_range': (0.6, 0.8),
                'vegetation_range': (0.4, 0.6),
                'soil_range': (0.7, 0.9),
                'preferred_soil': [1, 2],  # Loamy, Clayey
                'color': '#FF6347'  # Tomato red
            },
            'Maize': {
                'temp_range': (21, 27),
                'moisture_range': (0.6, 0.8),
                'vegetation_range': (0.5, 0.7),
                'soil_range': (0.6, 0.8),
                'preferred_soil': [1, 3],  # Loamy, Organic
                'color': '#FFD700'  # Gold
            }
        }

    def find_band_files(self, mtl_path: str) -> Dict[str, str]:
        """Find all band files associated with the MTL file"""
        base_dir = os.path.dirname(mtl_path)
        mtl_filename = os.path.basename(mtl_path)
        
        # Extract product ID from MTL filename (e.g., "LC08_L1TP_194056_20241221_20241228_02_T1" from "LC08_L1TP_194056_20241221_20241228_02_T1_MTL.txt")
        product_id = mtl_filename.split('_MTL')[0]
        
        bands = {}
        logger.info(f"Looking for band files in directory: {base_dir}")
        logger.info(f"Product ID: {product_id}")
        
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if os.path.isfile(file_path) and (file.endswith('.TIF') or file.endswith('.tif')):
                logger.info(f"Found TIF file: {file}")
                
                # Try to extract band name from filename
                # Look for patterns like "B1", "B2", "B3", etc. in the filename
                band_name = None
                
                # Method 1: Look for "B" followed by a number
                band_match = re.search(r'B(\d+)', file)
                if band_match:
                    band_name = f"B{band_match.group(1)}"
                else:
                    # Method 2: Look for band number in the filename
                    # Try to find patterns like "_B1_", "_B2_", etc.
                    for i in range(1, 12):  # Check for bands 1-11
                        if f"_B{i}_" in file or f"_B{i}." in file:
                            band_name = f"B{i}"
                            break
                
                if band_name:
                    bands[band_name] = file_path
                    logger.info(f"Added band {band_name}: {file}")
        
        logger.info(f"Found {len(bands)} band files: {list(bands.keys())}")
        return bands

    def read_band(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """Read a band file and return the data and profile"""
        with rasterio.open(file_path) as src:
            band = src.read(1).astype(float)
            profile = src.profile
        return band, profile

    # === Environmental Indices Calculations ===
    def compute_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index [-1 to 1]"""
        return np.clip((nir - red) / (nir + red + 1e-10), -1, 1)

    def compute_evi(self, nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Enhanced Vegetation Index [-1 to 1]"""
        return np.clip(2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1), -1, 1)

    def compute_savi(self, nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
        """Soil Adjusted Vegetation Index [-1 to 1]"""
        return np.clip((nir - red) / (nir + red + L) * (1 + L), -1, 1)

    def compute_ndmi(self, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """Normalized Difference Moisture Index [-1 to 1]"""
        return np.clip((nir - swir) / (nir + swir + 1e-10), -1, 1)

    def compute_ndbi(self, swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Built-up Index [-1 to 1]"""
        return np.clip((swir - nir) / (swir + nir + 1e-10), -1, 1)

    def compute_bsi(self, swir: np.ndarray, red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Bare Soil Index [-1 to 1]"""
        return np.clip(((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + 1e-10), -1, 1)

    def compute_carbon(self, ndvi: np.ndarray, lst: np.ndarray) -> np.ndarray:
        """Improved Carbon Storage Proxy [0 to 1]"""
        lst_norm = np.clip((lst - (-20)) / (60 - (-20)), 0, 1)
        return np.clip((1 - ndvi) * (1 - lst_norm), 0, 1)

    def compute_methane(self, ndvi: np.ndarray, ndbi: np.ndarray) -> np.ndarray:
        """Methane Emission Proxy [0 to 1] - Labeled as Urban Stress Index"""
        return np.clip((ndbi + (1 - ndvi)) / 2, 0, 1)

    def compute_moisture(self, ndvi: np.ndarray, ndmi: np.ndarray) -> np.ndarray:
        """Atmospheric Moisture Proxy [-1 to 1]"""
        return np.clip((ndvi + ndmi) / 2, -1, 1)

    def compute_soil_type(self, bsi: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
        """Classify soil type based on BSI and NDVI"""
        # 0: Sandy, 1: Loamy, 2: Clayey, 3: Organic
        soil_map = np.zeros_like(bsi)
        soil_map[(bsi > 0.5) & (ndvi < 0.2)] = 0  # Sandy
        soil_map[(bsi > -0.2) & (bsi <= 0.5) & (ndvi < 0.3)] = 1  # Loamy
        soil_map[(bsi <= -0.2) & (ndvi < 0.3)] = 2  # Clayey
        soil_map[ndvi >= 0.3] = 3  # Organic
        return soil_map

    def compute_lst(self, band10: np.ndarray, metadata_path: str) -> np.ndarray:
        """Land Surface Temperature (°C) with realistic bounds"""
        def extract_metadata_value(meta_file: str, key: str) -> float:
            with open(meta_file, 'r') as f:
                for line in f:
                    if key in line:
                        return float(line.split('=')[1].strip())
            raise ValueError(f"Metadata key {key} not found")
        
        try:
            ML = extract_metadata_value(metadata_path, 'RADIANCE_MULT_BAND_10')
            AL = extract_metadata_value(metadata_path, 'RADIANCE_ADD_BAND_10')
            K1 = extract_metadata_value(metadata_path, 'K1_CONSTANT_BAND_10')
            K2 = extract_metadata_value(metadata_path, 'K2_CONSTANT_BAND_10')
            
            radiance = ML * band10 + AL
            bt_kelvin = K2 / np.log((K1 / radiance) + 1)
            lst_celsius = bt_kelvin - 273.15
            lst_celsius = np.clip(lst_celsius, -20, 60)
            lst_celsius[np.abs(lst_celsius - np.mean(lst_celsius)) > 3*np.std(lst_celsius)] = np.nan
            return lst_celsius
        except Exception as e:
            logger.warning(f"Could not compute LST from metadata: {e}")
            # Fallback: use band10 directly as temperature proxy
            return np.clip(band10 * 0.1 - 273.15, -20, 60)

    def calculate_suitability(self, indices_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate suitability scores for all crops"""
        # Prepare normalized parameters (0-1 scale)
        lst_norm = (indices_dict['LST'] - (-20)) / (60 - (-20))
        moisture = (indices_dict['Moisture'] + 1) / 2
        vegetation = (indices_dict['NDVI'] + 1) / 2
        soil_quality = (indices_dict['Carbon'] + (1 - indices_dict['BSI'])) / 2
        
        # Get soil type compatibility
        soil_type = indices_dict['SoilType']
        
        suitability = {}
        for crop, params in self.crop_params.items():
            # Temperature suitability (Gaussian distribution)
            temp_center = np.mean(params['temp_range'])
            temp_width = (params['temp_range'][1] - params['temp_range'][0]) / 2
            temp_score = np.exp(-0.5 * ((lst_norm - temp_center/60) / (temp_width/60))**2)
            
            # Moisture suitability (triangular distribution)
            moisture_low, moisture_high = params['moisture_range']
            moisture_score = np.where(
                moisture < moisture_low,
                moisture / moisture_low,
                np.where(
                    moisture > moisture_high,
                    (1 - moisture) / (1 - moisture_high),
                    1
                )
            )
            
            # Vegetation suitability
            veg_low, veg_high = params['vegetation_range']
            veg_score = np.clip((vegetation - veg_low) / (veg_high - veg_low), 0, 1)
            
            # Soil suitability
            soil_low, soil_high = params['soil_range']
            soil_score = np.clip((soil_quality - soil_low) / (soil_high - soil_low), 0, 1)
            
            # Soil type compatibility
            soil_comp = np.isin(soil_type, params['preferred_soil']).astype(float)
            
            # Combined score with weights
            suitability[crop] = np.clip(
                (temp_score*0.3 + moisture_score*0.2 + veg_score*0.2 + soil_score*0.2 + soil_comp*0.1),
                0, 1
            )
        
        return suitability

    def generate_prediction(self, suitability: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> str:
        """Generate a prediction based on the suitability analysis"""
        # Calculate average suitability scores
        avg_scores = {crop: np.nanmean(score) for crop, score in suitability.items()}
        
        # Find most suitable crop
        best_crop = max(avg_scores.items(), key=lambda x: x[1])
        
        # Analyze environmental conditions
        lst_avg = np.nanmean(indices['LST'])
        moisture_avg = np.nanmean((indices['Moisture'] + 1) / 2)
        soil_type_counts = np.bincount(indices['SoilType'].flatten().astype(int))
        dominant_soil = np.argmax(soil_type_counts)
        soil_types = ['Sandy', 'Loamy', 'Clayey', 'Organic']
        
        # Generate prediction text
        prediction = f"""
        PREDICTION BASED ON ANALYSIS:
        
        The most suitable crop for this area is {best_crop[0]} with an average suitability score of {best_crop[1]:.2f}.
        
        Environmental Conditions:
        - Average Temperature: {lst_avg:.1f}°C
        - Average Moisture: {moisture_avg:.2f}
        - Dominant Soil Type: {soil_types[dominant_soil]}
        
        Recommendations:
        """
        
        # Add crop-specific recommendations
        if best_crop[0] == 'Cashew':
            prediction += "- Cashew trees thrive in warm climates with well-drained soils.\n"
            prediction += "- Ensure proper spacing (7-9m between trees) for optimal growth.\n"
        elif best_crop[0] == 'Cassava':
            prediction += "- Cassava grows well in various soil types but prefers sandy loams.\n"
            prediction += "- Plant cuttings at 1m spacing for optimal yield.\n"
        elif best_crop[0] == 'Tomatoes':
            prediction += "- Tomatoes require consistent moisture and fertile soil.\n"
            prediction += "- Consider drip irrigation for optimal water management.\n"
        elif best_crop[0] == 'Maize':
            prediction += "- Maize performs best in deep, well-drained soils with good organic matter.\n"
            prediction += "- Plant in rows with 75cm spacing for optimal growth.\n"
        
        # Add general agricultural advice
        prediction += "\nGeneral Agricultural Advice:\n"
        prediction += "- Conduct soil tests to verify nutrient levels before planting.\n"
        prediction += "- Consider crop rotation to maintain soil health.\n"
        prediction += "- Monitor weather forecasts for optimal planting and harvesting times.\n"
        
        return prediction

    def array_to_base64(self, array: np.ndarray, cmap: str = 'viridis') -> str:
        """Convert numpy array to base64 encoded image"""
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(array, cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close()
        
        # Convert to base64
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def get_statistics(self, array: np.ndarray, name: str) -> Dict[str, Any]:
        """Get statistics for an array"""
        valid_data = array[~np.isnan(array)]
        if len(valid_data) == 0:
            return {
                "name": name,
                "min": 0,
                "max": 0,
                "mean": 0,
                "percentile_25": 0,
                "percentile_75": 0
            }
        
        return {
            "name": name,
            "min": float(np.nanmin(array)),
            "max": float(np.nanmax(array)),
            "mean": float(np.nanmean(array)),
            "percentile_25": float(np.percentile(valid_data, 25)),
            "percentile_75": float(np.percentile(valid_data, 75))
        }

    async def analyze_from_mtl(self, mtl_path: str) -> Dict[str, Any]:
        """Run complete analysis from MTL file path. If no bands are found, return limited metadata-only analysis."""
        try:
            # Debug: List all files in the directory
            base_dir = os.path.dirname(mtl_path)
            logger.info(f"MTL file path: {mtl_path}")
            logger.info(f"Base directory: {base_dir}")
            logger.info("All files in directory:")
            for file in os.listdir(base_dir):
                file_path = os.path.join(base_dir, file)
                if os.path.isfile(file_path):
                    logger.info(f"  - {file} ({os.path.getsize(file_path)} bytes)")
            
            # Load and process bands
            bands = self.find_band_files(mtl_path)
            logger.info(f"Found {len(bands)} bands: {list(bands.keys())}")

            required_bands = {'B2': 'Blue', 'B4': 'Red', 'B5': 'NIR', 'B6': 'SWIR', 'B10': 'Thermal'}
            loaded_bands = {}

            # If no bands found, do limited analysis
            if not bands:
                logger.warning("No band files found, performing limited analysis")
                # Parse metadata from the .txt file
                metadata = {}
                band_references = {}
                calibration_constants = {}
                general_metadata = {}
                try:
                    with open(mtl_path, 'r') as f:
                        for line in f:
                            if '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"')
                                metadata[key] = value
                                # Group band references
                                if key.startswith('FILE_NAME_BAND') or key.startswith('FILE_NAME_'):
                                    band_references[key] = value
                                # Group calibration constants
                                elif 'REFLECTANCE_' in key or 'RADIANCE_' in key or 'TEMPERATURE_' in key or 'QUANTIZE_' in key:
                                    calibration_constants[key] = value
                                else:
                                    general_metadata[key] = value
                except Exception as e:
                    logger.warning(f"Could not parse metadata from {mtl_path}: {e}")
                return {
                    "status": "limited",
                    "message": "No band files found. Only metadata extracted. Full analysis requires band files (.TIF) in the same folder as the .txt file.",
                    "band_references": band_references,
                    "calibration_constants": calibration_constants,
                    "general_metadata": general_metadata,
                    "raw_metadata": metadata
                }

            logger.info("Loading required bands...")
            for band_code, band_name in required_bands.items():
                if band_code in bands:
                    loaded_bands[band_code], profile = self.read_band(bands[band_code])
                    logger.info(f"Loaded {band_name} band ({band_code})")
                else:
                    logger.error(f"Missing required band: {band_code} ({band_name})")
                    logger.error(f"Available bands: {list(bands.keys())}")
                    raise ValueError(f"Missing required band: {band_code} ({band_name})")
            
            # Calculate environmental indices
            logger.info("Calculating environmental indices...")
            ndvi = self.compute_ndvi(loaded_bands['B5'], loaded_bands['B4'])
            lst = self.compute_lst(loaded_bands['B10'], mtl_path)
            bsi = self.compute_bsi(loaded_bands['B6'], loaded_bands['B4'], loaded_bands['B5'], loaded_bands['B2'])
            
            indices = {
                'NDVI': ndvi,
                'EVI': self.compute_evi(loaded_bands['B5'], loaded_bands['B4'], loaded_bands['B2']),
                'SAVI': self.compute_savi(loaded_bands['B5'], loaded_bands['B4']),
                'NDMI': self.compute_ndmi(loaded_bands['B5'], loaded_bands['B6']),
                'NDBI': self.compute_ndbi(loaded_bands['B6'], loaded_bands['B5']),
                'BSI': bsi,
                'Carbon': self.compute_carbon(ndvi, lst),
                'Methane': self.compute_methane(ndvi, self.compute_ndbi(loaded_bands['B6'], loaded_bands['B5'])),
                'Moisture': self.compute_moisture(ndvi, self.compute_ndmi(loaded_bands['B5'], loaded_bands['B6'])),
                'LST': lst,
                'SoilType': self.compute_soil_type(bsi, ndvi)
            }
            
            # Perform crop suitability analysis
            logger.info("Assessing crop suitability...")
            suitability = self.calculate_suitability(indices)
            
            # Generate prediction
            prediction = self.generate_prediction(suitability, indices)
            
            # Prepare results
            env_stats = []
            for name, array in indices.items():
                if name != 'SoilType':
                    env_stats.append(self.get_statistics(array, name))
            
            crop_stats = []
            for crop in suitability:
                score = suitability[crop]
                crop_stats.append({
                    "crop": crop,
                    "min": float(np.nanmin(score)),
                    "max": float(np.nanmax(score)),
                    "mean": float(np.nanmean(score)),
                    "percentile_25": float(np.percentile(score[~np.isnan(score)], 25)),
                    "percentile_75": float(np.percentile(score[~np.isnan(score)], 75))
                })
            
            # Generate visualization images
            suitability_images = {}
            for crop, score in suitability.items():
                suitability_images[crop] = self.array_to_base64(score, 'RdYlGn')
            
            # Soil type visualization
            soil_colors = ['#F4E3AF', '#D2B48C', '#8B4513', '#556B2F']  # Sandy, Loamy, Clayey, Organic
            cmap_soil = LinearSegmentedColormap.from_list('soil_cmap', soil_colors, N=4)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            soil_plot = ax.imshow(indices['SoilType'], cmap=cmap_soil, vmin=0, vmax=3)
            ax.set_title("Soil Type Distribution")
            plt.colorbar(soil_plot, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plt.close()
            
            soil_image = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            
            return {
                "environmental_statistics": env_stats,
                "crop_suitability_statistics": crop_stats,
                "suitability_images": suitability_images,
                "soil_type_image": soil_image,
                "prediction": prediction,
                "best_crop": max(crop_stats, key=lambda x: x["mean"])["crop"],
                "analysis_summary": {
                    "total_pixels": int(ndvi.size),
                    "valid_pixels": int(np.sum(~np.isnan(ndvi))),
                    "bands_processed": list(loaded_bands.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Error during multispectral analysis: {str(e)}")
            raise e 