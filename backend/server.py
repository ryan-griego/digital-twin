def call_bedrock(conversation: List[Dict], user_message: str) -> str:
    """Call AWS Bedrock with conversation history"""

    # Build messages in Bedrock format
    messages = []

    # Correct system role
    messages.append({
        "role": "system",
        "content": [{"text": prompt()}]
    })

    # Optional: assistant acknowledgment
    messages.append({
        "role": "assistant",
        "content": [{"text": "Understood. I am Ryan's digital twin and will respond helpfully."}]
    })

    # Add history
    for msg in conversation[-10:]:
        messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })

    # Add new user message
    messages.append({
        "role": "user",
        "content": [{"text": user_message}]
    })

    try:
        response = bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 1500,
                "temperature": 0.7,
                "topP": 0.9
            }
        )

        # FIXED: Correct message retrieval
        return response["output"]["messages"][0]["content"][0]["text"]

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            raise HTTPException(status_code=400, detail="Invalid message format for Bedrock")
        elif error_code == 'AccessDeniedException':
            raise HTTPException(status_code=403, detail="Access denied to Bedrock model")
        else:
            raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")
