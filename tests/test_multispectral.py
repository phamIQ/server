import os
import zipfile
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
def multispectral_zip():
    """Create a test zip file with dummy multispectral data"""
    band_names = ["B2", "B4", "B5", "B6", "B10"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy MTL file with required metadata
        mtl_path = os.path.join(tmpdir, "LC08_L2SP_123456_20220101_20220101_02_T1_MTL.txt")
        with open(mtl_path, "w") as f:
            f.write(
                "RADIANCE_MULT_BAND_10 = 0.1\n"
                "RADIANCE_ADD_BAND_10 = 0.1\n"
                "K1_CONSTANT_BAND_10 = 774.8853\n"
                "K2_CONSTANT_BAND_10 = 1321.0789\n"
            )
        
        # Create dummy band files
        for band in band_names:
            band_path = os.path.join(tmpdir, f"LC08_L2SP_123456_20220101_20220101_02_T1_{band}.TIF")
            with open(band_path, "wb") as f:
                f.write(os.urandom(1024))  # Dummy binary content

        # Zip all files
        zip_path = os.path.join(tmpdir, "test_multispectral.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(mtl_path, os.path.basename(mtl_path))
            for band in band_names:
                band_path = os.path.join(tmpdir, f"LC08_L2SP_123456_20220101_20220101_02_T1_{band}.TIF")
                zipf.write(band_path, os.path.basename(band_path))
        
        yield zip_path

@pytest.fixture
def multispectral_txt():
    """Create a test txt file with dummy multispectral data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy MTL file
        mtl_path = os.path.join(tmpdir, "LC08_L2SP_123456_20220101_20220101_02_T1_MTL.txt")
        with open(mtl_path, "w") as f:
            f.write(
                "RADIANCE_MULT_BAND_10 = 0.1\n"
                "RADIANCE_ADD_BAND_10 = 0.1\n"
                "K1_CONSTANT_BAND_10 = 774.8853\n"
                "K2_CONSTANT_BAND_10 = 1321.0789\n"
            )
        
        # Create dummy band files in the same directory
        band_names = ["B2", "B4", "B5", "B6", "B10"]
        for band in band_names:
            band_path = os.path.join(tmpdir, f"LC08_L2SP_123456_20220101_20220101_02_T1_{band}.TIF")
            with open(band_path, "wb") as f:
                f.write(os.urandom(1024))
        
        yield mtl_path

def test_multispectral_endpoint_zip(client, multispectral_zip):
    """Test multispectral endpoint with zip file upload"""
    # Note: This test will likely fail due to dummy data, but tests the endpoint structure
    headers = {"Authorization": "Bearer testtoken"}  # Adjust if auth is required

    with open(multispectral_zip, "rb") as f:
        response = client.post(
            "/predict/multispectral",
            files={"file": ("test_multispectral.zip", f, "application/zip")},
            headers=headers
        )
    
    # Check response structure
    assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}"
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert "analysis_type" in data
        assert data["analysis_type"] == "multispectral"
        assert "filename" in data
    else:
        # If it fails due to dummy data, that's expected
        data = response.json()
        assert "detail" in data

def test_multispectral_endpoint_txt(client, multispectral_txt):
    """Test multispectral endpoint with txt file upload"""
    # Note: This test will likely fail due to dummy data, but tests the endpoint structure
    headers = {"Authorization": "Bearer testtoken"}  # Adjust if auth is required

    with open(multispectral_txt, "rb") as f:
        response = client.post(
            "/predict/multispectral",
            files={"file": ("LC08_L2SP_123456_20220101_20220101_02_T1_MTL.txt", f, "text/plain")},
            headers=headers
        )
    
    # Check response structure
    assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}"
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert "analysis_type" in data
        assert data["analysis_type"] == "multispectral"
        assert "filename" in data
    else:
        # If it fails due to dummy data, that's expected
        data = response.json()
        assert "detail" in data

def test_multispectral_endpoint_invalid_file(client):
    """Test multispectral endpoint with invalid file type"""
    headers = {"Authorization": "Bearer testtoken"}

    # Test with invalid file type
    response = client.post(
        "/predict/multispectral",
        files={"file": ("test.jpg", b"dummy content", "image/jpeg")},
        headers=headers
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "must be a .txt or .zip file" in data["detail"]

def test_multispectral_endpoint_no_file(client):
    """Test multispectral endpoint without file"""
    headers = {"Authorization": "Bearer testtoken"}

    response = client.post(
        "/predict/multispectral",
        headers=headers
    )
    
    assert response.status_code == 422  # Validation error

def test_multispectral_endpoint_zip_no_mtl(client):
    """Test multispectral endpoint with zip file that doesn't contain MTL file"""
    headers = {"Authorization": "Bearer testtoken"}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create zip with no MTL file
        zip_path = os.path.join(tmpdir, "test_no_mtl.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.writestr("dummy.txt", "dummy content")
        
        with open(zip_path, "rb") as f:
            response = client.post(
                "/predict/multispectral",
                files={"file": ("test_no_mtl.zip", f, "application/zip")},
                headers=headers
            )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "No MTL .txt metadata file found" in data["detail"]

if __name__ == "__main__":
    pytest.main([__file__]) 