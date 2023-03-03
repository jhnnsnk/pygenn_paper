for i in $(seq 0 9); do
    cat potjans_microcircuit_pygenn.templ | sed "s/__seed__/1234$i/" > potjans_microcircuit_pygenn.py
    python3 potjans_microcircuit_pygenn.py
    mv genn_simulation_data.json genn_simulation_data$i.json
    rm -rf potjans_microcircuit_CODE
done

