# README

## Running this experiment

### GitHub Codespaces

The simplest way to work with this experiment is to run it in GitHub Codespaces.
For this to work, you must first verify that:

1. This project is hosted in a GitHub repository;
2. This README file is located at the top level of that GitHub repository (i.e. not in a subdirectory).
3. You are logged into GitHub.

If the above requirements are satisfied, you can launch the experiment in Codespaces as follows.
On the repository page, click the green "Code" button, click "Codespaces",
and then click "Create codespace on main".
This should take you to a window that loads a VSCode environment for your repository.

It's not compulsory, but we recommend 'installing' this environment as a local app
(technically, a 'Progressive Web Application').
To do this in Chrome, go to your URL bar, and look on the right for an icon of a computer
with an arrow, which says 'Install' when you mouseover it. Click this, and your repository
should open in its own app. This provides a better UI experience, including better handling
of keyboard shortcuts.

The codespace will take a while to start up, because it needs to install the dependencies,
but don't worry, this is a one-time process.
Once the codespace is ready, you can then launch the experiment in debug mode by running the
following terminal command:

```bash
psynet debug local
```

Wait a moment, and then a browser window should open containing a link to the dashboard.
Click it, then enter 'admin' as both username and password, then press OK.
You'll now see the experiment dashboard.
Click 'Development', then 'New participant', to create a link to try the experiment
as a participant.

### Locally in a virtual environment

A more conventional approach is to instead run this demo locally in a virtual environment.
This is more involved as you have to install several related dependencies like Redis and PostgreSQL.
To do so, navigate to the [PsyNet website](https://psynet.dev) and follow the 'virtual environment'
installation instructions. Check `constraints.txt` to find the recommended Python version
for this experiment.

### Other options

It should also be possible to load this repository using Devcontainers in an IDE such as VSCode.
In theory, this should function equivalently to GitHub Codespaces. However, this hasn't worked
so reliably for us yet, and we're still figuring out how to make it work better.
