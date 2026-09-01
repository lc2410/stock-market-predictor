from app import app


def test_app_initialization():
    """Test that the Flask app initializes correctly."""
    assert app is not None
    assert app.name == "app"

def test_blueprints_registered():
    """Test that the blueprints are registered correctly."""
    blueprints = app.blueprints
    assert "search" in blueprints
    assert "prediction" in blueprints
    assert "screener" in blueprints
