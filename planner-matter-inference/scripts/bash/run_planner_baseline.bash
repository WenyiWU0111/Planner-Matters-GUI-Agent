domains=(Amazon Allrecipes Coursera Google_Map)
# domains=(wikipedia)
# domains=(test_domain_Info test_domain_Service test_website)

for domain in "${domains[@]}"; do
    echo "Running inference for $domain"
    
    # Set eval_type based on domain
    if [ "$domain" = "wikipedia" ]; then
        eval_type="mmina"
    elif [[ "$domain" == "test_domain_Info" || "$domain" == "test_domain_Service" || "$domain" == "test_website" ]]; then
        eval_type="mind2web_executable"
    else
        eval_type="webvoyager"
    fi
    
    python run.py --domain "$domain" \
        --evaluation_type "$eval_type" \
        --model qwen2.5-vl \
        --datetime memory_planner \
        --use_planner_with_memory \
        --planner_server_url http://localhost:8000/v1 \
        --planner_model Qwen/Qwen2.5-VL-7B-Instruct \
        --use_adaptive_memory \
        --faiss_index_path "${FAISS_INDEX_PATH:-memory_index/simple_text}" \
        --memory_data_dir "${MEMORY_DATA_DIR:-data/trajectories}"
        
done
