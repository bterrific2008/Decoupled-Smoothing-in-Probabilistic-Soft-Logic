#!/bin/bash
# Run a decoupled smoothing method on a single data variation

readonly THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DATA_DIR="${THIS_DIR}/data"
readonly BASE_OUT_DIR="${THIS_DIR}/results"

readonly ADDITIONAL_PSL_OPTIONS='--int-ids -D random.seed=12345 -D log4j.threshold=debug -D log4j.threshold=TRACE --postgres btor'
readonly ADDITIONAL_LEARN_OPTIONS='--learn GaussianProcessPrior -D weightlearning.evaluator=RankingEvaluator -D rankingevaluator.representative=AUROC'
readonly ADDITIONAL_EVAL_OPTIONS='--infer --eval CategoricalEvaluator RankingEvaluator'

# An identifier to differentiate the output of this script/experiment from other scripts.
readonly RUN_ID='decoupled-smoothing'

function display_help() {
  echo "USAGE: $0 <data> <random seed> <percent labeled> {learn|eval} <method dir> ..."
  exit 1
}

function generate_data() {
  random_seed=$(printf "%04d" $1)
  data_name=$2
  train_test=$3

  local logPath="${BASE_DATA_DIR}/${train_test}/${data_name}/01pct/${random_seed}rand/data_log.json"
  echo "Log path: ${logPath}"

  if [[ -e "${logPath}" ]]; then
    echo "Output data already exists, skipping data generation"
  fi
  if [[ "$train_test" = learn ]]; then
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 write_psl_data_snowball.py --seed ${random_seed} --data ${data_name}.mat --learn --closefriend
  else
    echo "Generating data with seed ${random_seed} and data ${data_name} for ${train_test}"
    python3 write_psl_data_snowball.py --seed ${random_seed} --data ${data_name}.mat
  fi
}

function main() {
  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <data name>"
    exit 1
  fi

  data_name=$1

  trap exit SIGINT

  generate_data "4212" "${data_name}" "learn"

  for rand_sd in 1 12345 837 2841 4293 6305 6746 9056 9241 9547; do
    generate_data "${rand_sd}" "${data_name}" "eval"
  done

  return 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
