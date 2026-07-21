import pytest
from plugins.plugin_registry import PluginRegistry
from plugins.filesystem_plugin.filesystem_plugin import FilesystemPlugin
from plugins.terminal_plugin.terminal_plugin import TerminalPlugin

@pytest.mark.asyncio
async def test_filesystem_plugin(tmp_path):
    plugin = FilesystemPlugin()
    assert plugin.name() == "filesystem"
    
    file_path = str(tmp_path / "sample.txt")
    write_res = await plugin.run("write_file", {"path": file_path, "content": "Hello JARVIS!"})
    assert write_res["status"] == "written"
    
    read_res = await plugin.run("read_file", {"path": file_path})
    assert read_res["content"] == "Hello JARVIS!"

@pytest.mark.asyncio
async def test_plugin_registry():
    registry = PluginRegistry()
    registry.register(FilesystemPlugin())
    registry.register(TerminalPlugin())
    
    plugins = registry.list_plugins()
    assert len(plugins) == 2
    assert registry.get("filesystem") is not None
