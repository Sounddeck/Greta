# Neovim Integration for Greta PAI
# Text editor integration for coding workflows

import os
import json
import asyncio
import tempfile
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class NeovimConnector:
    """Neovim integration for Greta PAI - connects AI assistance with coding workflows"""

    def __init__(self, nvim_socket: Optional[str] = None):
        self.nvim_socket = nvim_socket or os.getenv("NVIM_LISTEN_ADDRESS")
        self.temp_dir = Path(tempfile.gettempdir()) / "greta_neovim"
        self.temp_dir.mkdir(exist_ok=True)

        # Initialize integration
        self._initialize_neovim_integration()

    def _initialize_neovim_integration(self):
        """Set up Neovim integration files and configurations"""

        # Create Greta PAI nvim plugin
        plugin_content = self._create_nvim_plugin()
        lua_config = self._create_lua_config()
        keymaps = self._create_keymaps()

        # Save integration files
        self._save_integration_files(plugin_content, lua_config, keymaps)

    def _create_nvim_plugin(self) -> str:
        """Create the main Neovim plugin file"""
        return '''
-- Greta PAI Integration for Neovim
-- AI-assisted coding workflow integration

local M = {}

-- Configuration
M.config = {
    server_url = "http://localhost:8000/api/v1",
    enable_ai_suggestions = true,
    enable_code_analysis = true,
    enable_auto_complete = true,
    update_interval = 3000, -- milliseconds
    max_context_lines = 50
}

-- State
M.current_context = {}
M.last_analysis = nil
M.session_id = os.time()

-- Core Functions

function M.setup(opts)
    if opts then
        M.config = vim.tbl_extend("force", M.config, opts)
    end
    M._setup_autocommands()
    M._setup_commands()
    print("🤖 Greta PAI Integration Loaded!")
end

function M._setup_autocommands()
    vim.api.nvim_create_augroup("GretaPAI", { clear = true })

    -- Auto-analyze current buffer on save
    vim.api.nvim_create_autocmd("BufWritePost", {
        group = "GretaPAI",
        callback = M.analyze_current_buffer
    })

    -- Real-time suggestions on cursor move
    vim.api.nvim_create_autocmd("CursorMoved", {
        group = "GretaPAI",
        callback = M.update_suggestions
    })

    -- Code analysis on file open
    vim.api.nvim_create_autocmd("BufReadPost", {
        group = "GretaPAI",
        callback = M.analyze_file
    })
end

function M._setup_commands()
    -- Greta PAI commands
    vim.api.nvim_create_user_command("GretaAnalyze", M.analyze_current_buffer, {})
    vim.api.nvim_create_user_command("GretaSuggest", M.show_suggestions, {})
    vim.api.nvim_create_user_command("GretaRefactor", M.refactor_selection, {})
    vim.api.nvim_create_user_command("GretaOptimize", M.optimize_code, {})
    vim.api.nvim_create_user_command("GretaDocument", M.generate_documentation, {})
    vim.api.nvim_create_user_command("GretaTest", M.generate_tests, {})
    vim.api.nvim_create_user_command("GretaDebug", M.debug_code, {})
end

-- Core AI Features

function M.analyze_current_buffer()
    local current_file = vim.api.nvim_buf_get_name(0)
    local content = M._get_buffer_content()

    if M.config.enable_code_analysis then
        M._send_to_greta("analyze_code", {
            filename = current_file,
            content = content,
            language = M._detect_language()
        })
    end
end

function M.show_suggestions()
    local context = M._get_cursor_context()
    M._send_to_greta("get_suggestions", {
        context = context,
        language = M._detect_language()
    })
end

function M.refactor_selection()
    local start_pos = vim.api.nvim_buf_get_mark(0, "<")
    local end_pos = vim.api.nvim_buf_get_mark(0, ">")

    if start_pos and end_pos then
        local lines = vim.api.nvim_buf_get_lines(0, start_pos[1]-1, end_pos[1], false)
        local refactored_code = M._send_to_greta_sync("refactor_code", {
            code = table.concat(lines, "\\n"),
            language = M._detect_language()
        })

        if refactored_code then
            vim.api.nvim_buf_set_lines(0, start_pos[1]-1, end_pos[1], false, vim.split(refactored_code, "\\n"))
        end
    end
end

function M.optimize_code()
    local content = M._get_buffer_content()
    local optimized = M._send_to_greta_sync("optimize_code", {
        code = content,
        language = M._detect_language()
    })

    if optimized then
        vim.api.nvim_buf_set_lines(0, 0, -1, false, vim.split(optimized, "\\n"))
    end
end

function M.generate_documentation()
    local content = M._get_buffer_content()
    local docs = M._send_to_greta_sync("generate_docs", {
        code = content,
        language = M._detect_language()
    })

    if docs then
        M._open_documentation_buffer(docs)
    end
end

function M.generate_tests()
    local content = M._get_buffer_content()
    local tests = M._send_to_greta_sync("generate_tests", {
        code = content,
        language = M._detect_language()
    })

    if tests then
        M._create_test_file(tests)
    end
end

function M.debug_code()
    local content = M._get_buffer_content()
    local issues = M._send_to_greta_sync("find_bugs", {
        code = content,
        language = M._detect_language()
    })

    if issues then
        M._show_debug_quickfix(issues)
    end
end

-- Helper Functions

function M._get_buffer_content()
    return table.concat(vim.api.nvim_buf_get_lines(0, 0, -1, false), "\\n")
end

function M._get_cursor_context()
    local row, col = unpack(vim.api.nvim_win_get_cursor(0))
    local start_row = math.max(1, row - M.config.max_context_lines)
    local end_row = math.min(vim.api.nvim_buf_line_count(0), row + M.config.max_context_lines)
    local lines = vim.api.nvim_buf_get_lines(0, start_row-1, end_row, false)
    return table.concat(lines, "\\n")
end

function M._detect_language()
    local filetype = vim.bo.filetype or ""
    return filetype
end

function M._send_to_greta(endpoint, data)
    vim.defer_fn(function()
        -- Asynchronous request to Greta PAI
        local cmd = string.format(
            "curl -s -X POST -H 'Content-Type: application/json' -d '%s' %s/api/v1/%s",
            vim.fn.json_encode(data), M.config.server_url, endpoint
        )
        local response = vim.fn.system(cmd)
        M._handle_response(response, endpoint)
    end, 0)
end

function M._send_to_greta_sync(endpoint, data)
    local cmd = string.format(
        "curl -s -X POST -H 'Content-Type: application/json' -d '%s' %s/api/v1/%s",
        vim.fn.json_encode(data), M.config.server_url, endpoint
    )
    return vim.fn.system(cmd)
end

function M._handle_response(response, endpoint)
    if response ~= "" then
        local parsed = vim.fn.json_decode(response)
        if parsed and parsed.success then
            M._display_results(endpoint, parsed.data)
        end
    end
end

function M._display_results(endpoint, data)
    if endpoint == "analyze_code" then
        M._show_code_analysis(data)
    elseif endpoint == "get_suggestions" then
        M._show_suggestions(data)
    else
        vim.notify("Greta PAI: " .. endpoint .. " completed")
    end
end

function M._show_code_analysis(data)
    -- Display code analysis results
    local qf_list = {}
    for _, issue in ipairs(data.issues or {}) do
        table.insert(qf_list, {
            filename = vim.api.nvim_buf_get_name(0),
            lnum = issue.line or 1,
            col = issue.col or 1,
            text = issue.message or "",
            type = issue.severity == "error" and "E" or "W"
        })
    end

    if #qf_list > 0 then
        vim.fn.setqflist(qf_list)
        vim.cmd("copen")
    end
end

function M._show_suggestions(data)
    -- Display AI suggestions
    local suggestions = data.suggestions or {}
    if #suggestions == 0 then return end

    local lines = {}
    table.insert(lines, "🤖 Greta PAI Suggestions:")
    table.insert(lines, "")

    for _, suggestion in ipairs(suggestions) do
        table.insert(lines, "• " .. suggestion.text)
    end

    -- Create or update suggestion buffer
    M._show_popup_buffer("GretaSuggestions", lines)
end

function M._show_popup_buffer(buf_name, content)
    -- Create floating window with suggestions
    local buf = vim.api.nvim_create_buf(false, true)
    vim.api.nvim_buf_set_name(buf, buf_name)
    vim.api.nvim_buf_set_lines(buf, 0, -1, false, content)

    local width = 50
    local height = 15
    local opts = {
        relative = 'cursor',
        width = width,
        height = height,
        col = 1,
        row = 1,
        style = 'minimal',
        border = 'rounded'
    }

    local win = vim.api.nvim_open_win(buf, true, opts)
    vim.cmd("autocmd WinLeave <buffer=" .. buf .. "> :q!")
end

function M._open_documentation_buffer(docs)
    -- Open documentation in split
    vim.cmd("vsplit")
    vim.cmd("enew")
    vim.api.nvim_buf_set_lines(0, 0, -1, false, vim.split(docs, "\\n"))
    vim.bo.filetype = "markdown"
end

function M._create_test_file(tests)
    local current_file = vim.api.nvim_buf_get_name(0)
    local test_file = M._get_test_filename(current_file)

    -- Create or update test file
    vim.cmd("edit " .. test_file)
    vim.api.nvim_buf_set_lines(0, 0, -1, false, vim.split(tests, "\\n"))
end

function M._show_debug_quickfix(issues)
    local qf_items = {}
    for _, issue in ipairs(issues.bugs or {}) do
        table.insert(qf_items, {
            filename = vim.api.nvim_buf_get_name(0),
            lnum = issue.line or 1,
            text = issue.description or "",
            type = "E"
        })
    end

    vim.fn.setqflist(qf_items)
    vim.cmd("copen")
end

function M._get_test_filename(filename)
    -- Generate test file name based on current file
    local base_name = filename:gsub("\\.[^.]*$", "")
    return base_name .. "_test.py"  -- Assuming Python, could be adapted
end

return M
'''

    def _create_lua_config(self) -> str:
        """Create Lua configuration file"""
        return '''
-- Greta PAI Neovim Configuration

-- Load Greta PAI plugin
require('greta').setup({
    server_url = os.getenv("GRETA_NEURO_URL") or "http://localhost:8000/api/v1",
    enable_ai_suggestions = true,
    enable_code_analysis = true,
    enable_auto_complete = true,
    update_interval = 3000,
    max_context_lines = 50
})

-- Custom keybindings for Greta PAI features
vim.keymap.set('n', '<leader>ga', ':GretaAnalyze<CR>', { desc = 'Analyze code with Greta PAI' })
vim.keymap.set('n', '<leader>gs', ':GretaSuggest<CR>', { desc = 'Show AI suggestions' })
vim.keymap.set('v', '<leader>gr', ':GretaRefactor<CR>', { desc = 'Refactor selected code' })
vim.keymap.set('n', '<leader>go', ':GretaOptimize<CR>', { desc = 'Optimize current file' })
vim.keymap.set('n', '<leader>gd', ':GretaDocument<CR>', { desc = 'Generate documentation' })
vim.keymap.set('n', '<leader>gt', ':GretaTest<CR>', { desc = 'Generate tests' })
vim.keymap.set('n', '<leader>gb', ':GretaDebug<CR>', { desc = 'Find bugs' })

-- Status line integration
vim.cmd([[
function! GretaPAIStatus()
    if exists('b:greta_analysis')
        return '🤖'
    else
        return ''
    endif
endfunction

set statusline+=%{GretaPAIStatus()}
]])
'''

    def _create_keymaps(self) -> str:
        """Create keymapping configuration"""
        return '''
" Greta PAI Neovim Keymaps
" Fast AI assistance keys

" Leader key for Greta commands (default \\)
nmap <leader>ga :GretaAnalyze<CR>
nmap <leader>gs :GretaSuggest<CR>
vmap <leader>gr :GretaRefactor<CR>
nmap <leader>go :GretaOptimize<CR>
nmap <leader>gd :GretaDocument<CR>
nmap <leader>gt :GretaTest<CR>
nmap <leader>gb :GretaDebug<CR>

" Quick AI help
nmap K :GretaSuggest<CR>
'''

    def _save_integration_files(self, plugin: str, lua_config: str, keymaps: str):
        """Save all integration files to appropriate locations"""

        # Create Neovim config directory
        config_dir = self.temp_dir / "neovim_config"
        config_dir.mkdir(exist_ok=True)

        # Save plugin
        plugin_file = config_dir / "greta.lua"
        plugin_file.write_text(plugin)

        # Save Lua config
        lua_file = config_dir / "init.lua"
        lua_file.write_text(lua_config)

        # Save keymaps
        keymap_file = config_dir / "keymaps.vim"
        keymap_file.write_text(keymaps)

    def get_neovim_integration_path(self) -> Path:
        """Get path to Neovim integration files"""
        return self.temp_dir / "neovim_config"

    def setup_neovim_config(self) -> str:
        """Generate setup instructions for Neovim integration"""
        return f'''
# Greta PAI Neovim Integration Setup

1. Copy the integration files to your Neovim config:
   cp {self.get_neovim_integration_path()}/* ~/.config/nvim/lua/

2. Add to your init.lua:
   require('greta')

3. Optional: Add keymap settings to your init.lua:
   require('keymaps')

4. Set environment variable:
   export GRETA_NEOVIM_URL="http://localhost:8000/api/v1"

5. Restart Neovim or run:
   :luafile ~/.config/nvim/init.lua

## Available Commands:
- :GretaAnalyze - Analyze current file
- :GretaSuggest - AI coding suggestions
- :GretaRefactor - Refactor selected code
- :GretaOptimize - Optimize code performance
- :GretaDocument - Generate documentation
- :GretaTest - Generate unit tests
- :GretaDebug - Find programming bugs

## Keybindings (leader key = \\):
- \\ga - Analyze code
- \\gs - Show suggestions
- K - Quick suggestion (when cursor on ambiguous code)
'''

    def test_integration(self) -> Dict[str, Any]:
        """Test the Neovim integration connectivity"""
        test_payload = {
            "language": "python",
            "code": "print('hello world')"
        }

        # Mock test - in real implementation would connect to Greta server
        return {
            "status": "connected",
            "server_url": self._get_server_url(),
            "integration_ready": True
        }

    def _get_server_url(self) -> str:
        """Get Greta PAI server URL"""
        return os.getenv("GRETA_NEOVIM_URL", "http://localhost:8000/api/v1")


def main():
    """CLI interface for Neovim integration"""
    import argparse

    parser = argparse.ArgumentParser(description="Greta PAI Neovim Integration")
    parser.add_argument("command", choices=["setup", "test", "path"],
                       help="Command to execute")

    args = parser.parse_args()

    connector = NeovimConnector()

    if args.command == "setup":
        print(connector.setup_neovim_config())

    elif args.command == "test":
        result = connector.test_integration()
        print(json.dumps(result, indent=2))

    elif args.command == "path":
        print(str(connector.get_neovim_integration_path()))


if __name__ == "__main__":
    main()
