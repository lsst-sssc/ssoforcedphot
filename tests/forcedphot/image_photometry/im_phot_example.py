from astropy.time import Time
import time
import json
from dataclasses import asdict
from forcedphot.image_photometry.photometry_service import PhotometryService
from forcedphot.image_photometry.utils import SearchParameters
from forcedphot.image_photometry.image_service import ImageService

def main():
    start_time = time.time()
    params = SearchParameters(
        bands={'r', 'i', 'g''u', 'z', 'y'},
        # bands={'r'},
        # ephemeris_file='./hygiea10_eph2.ecsv'
        ephemeris_file='./data/test_ephemeris_for_imphot.ecsv'

    )
    
    image_service = ImageService()
    image_metadata = image_service.search_images(params)

    if image_metadata is None:
        return print("The image search was unsuccesfull.")
    
    # Initialize the photometry service
    phot_service = PhotometryService(detection_threshold=5)
    results=[]
    for i in range(len(image_metadata)):
        
        result = phot_service.process_image(
            image_metadata=image_metadata[i] ,
            target_name="Example Target",
            target_type="smallbody",
            ephemeris_service="JPL Horizons",
            cutout_size=800,
            save_cutout=False,
            display=False,
            output_dir="./data",
            save_json=False,
            json_filename="test_output.json"
        )
        results.append(result)
    
    # `dataclass`-ok konvertálása JSON-kompatibilis formátumra
    json_data = [asdict(obj) for obj in results]
    
    # Save results to a JSON file
    with open("./data/test_output.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2, default=str)
    
    print(f"Total time: {time.time() - start_time:.2f} s")

    # Print results
    print("\nTarget Information:")
    print(f"Name: {result.target_name}")
    print(f"Type: {result.target_type}")
    print(f"Band: {result.band}")
    print(f"Visit ID: {result.visit_id}")
    print(f"Detector ID: {result.detector_id}")

    for i in range(len(results)):
        print("-" * 70)
        if results[i].forced_phot_on_target:
            print(f"\n{i} Target Photometry:")
            phot = results[i].forced_phot_on_target
            print(f"RA, Dec: ({phot.ra}°, {phot.dec}°)")
            print(f"Magnitude: {phot.mag:.2f} ± {phot.mag_err:.2f}")
            print(f"SNR: {phot.snr:.2f}")
            print(f"Flux: {phot.flux:.2f} ± {phot.flux_err:.2f}")
        
        if results[i].phot_within_error_ellipse:
            print(f"\nSources within error ellipse: {len(results[i].phot_within_error_ellipse)}")
            for j, source in enumerate(results[i].phot_within_error_ellipse, 1):
                print(f"\nSource {j}:")
                print(f"RA, Dec: ({source.ra}°, {source.dec}°)")
                print(f"Magnitude: {source.mag:.2f} ± {source.mag_err:.2f}")
                print(f"SNR: {source.snr:.2f}")
                print(f"Separation: {source.separation:.2f} arcsec")
                print(f"Sigma: {source.sigma:.2f} ")


if __name__ == "__main__":
    main()
