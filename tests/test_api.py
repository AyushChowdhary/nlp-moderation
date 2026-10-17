from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "NLP Moderation API is running"
    }


def test_empty_text_is_rejected():

    response = client.post(
        "/moderate",
        json={"text": "   "}
    )

    assert response.status_code == 400

    assert response.json()["detail"] == "Text cannot be empty"


def test_missing_text_is_rejected():

    response = client.post(
        "/moderate",
        json={}
    )

    assert response.status_code == 422


def test_text_length_limit():

    response = client.post(
        "/moderate",
        json={"text": "a" * 5001}
    )

    assert response.status_code == 422


def test_response_contains_process_time_header():

    response = client.get("/")

    assert response.status_code == 200

    assert "X-Process-Time" in response.headers
    