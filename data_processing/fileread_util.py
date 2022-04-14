#!/usr/bin/python
# encoding: utf-8


def txt_parse(f):
    result = []
    with open(f) as fp:
        line = fp.readline()
        result.append(line)
        while line:
            line = fp.readline()
            result.append(line)
    return result

