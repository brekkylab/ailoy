from pytest import Parser


def pytest_addoption(parser: Parser):
    parser.addoption(
        "-l",
        "--langmodel",
        action="store",
        choices=[
            "qwen3",
            "openai",
            "gemini",
            "claude",
            "grok",
        ],
        default="qwen3",
        help="LangModel provider",
    )
