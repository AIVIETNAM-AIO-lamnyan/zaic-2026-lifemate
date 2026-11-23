#!/usr/bin/env bash

load_lifemate_config() {
    local project_root="$1"
    local config_line
    local encoded_value
    local setting_name
    local setting_value
    local config_output

    config_output="$(python "$project_root/src/lifemate_config.py" --shell --config "$project_root/config.toml")" || return 1
    while IFS= read -r config_line; do
        setting_name="${config_line%%=*}"
        encoded_value="${config_line#*=}"
        setting_value="$(printf '%s' "$encoded_value" | base64 --decode)" || return 1
        export "$setting_name=$setting_value"
    done <<< "$config_output"
}
