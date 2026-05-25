# frozen_string_literal: true

source "https://rubygems.org"
gemspec

gem "jekyll", ENV["JEKYLL_VERSION"] if ENV["JEKYLL_VERSION"]
gem "kramdown-parser-gfm" if ENV["JEKYLL_VERSION"] == "~> 3.9"

# Ensure plugins required by _config.yml are available when using Bundler
gem "jekyll-feed", "~> 0.9"
gem "jekyll-seo-tag", "~> 2.1"
gem "jekyll-remote-theme"
