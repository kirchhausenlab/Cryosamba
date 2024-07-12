#!/bin/bash
install_requirements(){
    pip3 install pipreqs
    pipreqs ../../.
}
install_requirements