#!/usr/bin/bash

set -e
shopt -s globstar

while getopts p flag
do
	case "${flag}" in
		p) push=true;;
	esac
done

# Merge
if [ $push ]; then
	git checkout docs
	git merge master -m "🔀 Merge master into docs"
fi

# Generate HTML
sphinx-build -b html . ../docs

# Remove auto-generated RST files
rm -r api

# Edit HTML
cd ../docs
sed "s|<span class=\"pre\">\[source\]</span>|<i class=\"fa-solid fa-code\"></i>|g" -i **/*.html
sed "s|\(<a class=\"reference external\".*</a>\)\(<a class=\"headerlink\".*</a>\)|\2\1|g" -i **/*.html
sed "s|<a class=\"muted-link\" href=\"https://pradyunsg.me\">@pradyunsg</a>'s||g" -i **/*.html

# Push
if [ $push ]; then
	git add .
	git commit -m "📝 Update Sphinx documentation"
fi
