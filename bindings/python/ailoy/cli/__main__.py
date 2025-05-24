import typer

from ailoy.cli.model import app as model_app

app = typer.Typer(no_args_is_help=True)

app.add_typer(model_app, name="model")


if __name__ == "__main__":
    app()
