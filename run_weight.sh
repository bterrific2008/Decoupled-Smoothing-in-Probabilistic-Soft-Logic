#!/bin/bash
# Run a decoupled smoothing method against all data variations.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# An identifier to differentiate the output of this script/experiment from other scripts.
# readonly RUN_ID='run-all-weight-decoupled-smoothing'

function main() {
  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <data name> <method cli_dir>"
    exit 1
  fi

  data_name=$1
  method=$1

  trap exit SIGINT

  # eval the data
  for pct_lbl in 01 05 10 20 30 40 50 60 70 80 90 95; do
    for sub_method in method; do
      ./run_method.sh "${data_name}" "4212" "${pct_lbl}" "learn" "${sub_method}"

      for rand_sd in 1 12345 837 2841 4293 6305 6746 9056 9241 9547; do
        ./run_method.sh "${data_name}" "${rand_sd}" "${pct_lbl}" "eval" "${sub_method}"
      done
    done
  done

  return 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
