#! /bin/bash

cd "$(dirname "$(readlink -f "$0")")"

for t in normal no_encoder no_Attention no_GCN
do
    python ../model/main.py --dataset mESC --itr 5 --test $t --result_path ../test/model_test/$t
done

echo -e "\nNEXT, YOU MAY RUN \`scripts/04_velocity.Rmd\` or\n                  \`scripts/07_gene_module.ipynb\` or\n                  \`scripts/modelComp.Rmd\` TO DO DOWNSTREAM ANALYSIS\n"
