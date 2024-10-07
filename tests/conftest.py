import pytest

@pytest.fixture(scope="module")
def image_file():
    with open("test.jpeg", "rb") as image:
        yield image.read()

@pytest.fixture(scope="module")
def image_file_gun6():
    with open("gun6.jpg", "rb") as image:
        yield image.read()
