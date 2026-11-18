"""Dev-only support of the black-box harness: the fakes' logic
and servers, the harness's client side and helpers, the config overlay and the transcript redaction. Installed in the
workspace's test environment and in the harness's fakes image only: import-linter's testing-dev-only contract keeps
packages/ and apps/ from importing it, and no service image installs it."""
