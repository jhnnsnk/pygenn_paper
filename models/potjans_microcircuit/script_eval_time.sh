for i in $(seq 0 9); do
    cat eval_potjans_microcircuit_pygenn.templ | sed "s/__seed__/1234$i/" > eval_potjans_microcircuit_pygenn.py
    python3  eval_potjans_microcircuit_pygenn.py
    mv eval_genn_simulation_data.json eval_genn_simulation_data$i.json
    rm -rf potjans_microcircuit_CODE
done
