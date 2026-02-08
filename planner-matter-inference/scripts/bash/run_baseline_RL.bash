# domains=(Amazon Allrecipes Coursera Google_Map)
# domains=(wikipedia)
domains=(test_domain_Info test_domain_Service test_website)
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
        --model base-rl-qwen3 \
        --datetime rl-baseline \
        --checkpoint_path "${CHECKPOINT_PATH}"
        
done

# --open_router_api_key "${OPEN_ROUTER_API_KEY}"