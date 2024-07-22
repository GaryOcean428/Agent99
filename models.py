def get_model_info(model_key):
    return (
        {
            "1": {"name": "sonnet-3.5", "id": "claude-3-5-sonnet-20240620"},
            "2": {"name": "opus-3", "id": "claude-3-opus-20240229"},
            "3": {"name": "sonnet-3", "id": "claude-3-sonnet-20240229"},
            "4": {"name": "haiku-3", "id": "claude-3-haiku-20240307"},
        }
    ).get(model_key)


def get_model_list():

    return {
        "1": {"name": "sonnet-3.5", "id": "claude-3-5-sonnet-20240620"},
        "2": {"name": "opus-3", "id": "claude-3-opus-20240229"},
        "3": {"name": "sonnet-3", "id": "claude-3-sonnet-20240229"},
        "4": {"name": "haiku-3", "id": "claude-3-haiku-20240307"},
    }
