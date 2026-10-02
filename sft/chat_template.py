BEGIN_OF_TEXT = 1
START_HEADER = 2
END_HEADER = 3
EOT = 4

def encode(text):
    return [ord(c) for c in text]
def role_header(role):
    return (
        [START_HEADER]
        + encode(role)
        + [END_HEADER]
        + encode("\n\n")
    )
def encode_chat(user_message: str, system_message: str = None) -> list:
    if system_message is None:
        system_message="You are a helpful assistant."
    return (
        [BEGIN_OF_TEXT]
        + role_header("system")
        + encode(system_message)
        + [EOT]
        + role_header("user")
        + encode(user_message)
        + [EOT]
        + role_header("assistant")
    )