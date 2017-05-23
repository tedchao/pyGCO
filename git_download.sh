#!/usr/bin/sh
cd gco_source
git init
git remote add origin https://github.com/kayarre/gco-v3.0.git
git fetch
git checkout dev
cd ..
