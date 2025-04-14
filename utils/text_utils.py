def text_to_task_list(text_descriptions):
    """
    Convert text descriptions to a task list
    Input format can be either:
    1. Single text string:
    Event Type: Low Light
    Description: This infrared image suffers from low light degradation...
    
    2. List of text strings (batch processing):
    [
        "Event Type: Low Light\nDescription: This infrared image...",
        "Event Type: Haze\nDescription: Another description..."
    ]
    
    Returns: 
    For single text: [{
        'event': 'low_light', 
        'description': '...'
    }]
    
    For batch: [
        [{
            'event': 'low_light', 
            'description': '...'
        }],
        [{
            'event': 'haze',
            'description': '...'
        }]
    ]
    """
    if isinstance(text_descriptions, list):
        return [_process_single_text(text) for text in text_descriptions]
    
    return _process_single_text(text_descriptions)

def _process_single_text(text):
    task_list = []
    
    lines = text.strip().split('\n')
    
    for i in range(0, len(lines), 2):
        event_line = lines[i].strip()
        desc_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        
        if event_line.startswith('Event Type:'):
            event = event_line.replace('Event Type:', '').strip().lower()
            event = event.replace(' ', '_')
            
            description = desc_line.replace('Description:', '').strip()
            
            task = {
                'event': event,
                'description': description
            }
            task_list.append(task)
    
    return task_list

def truncate_text_batch(text_batch):
    """
    Process a batch of texts, ensuring each text does not exceed CLIP's context length (77 tokens).
    
    Args:
        text_batch: List[str] Each string is formatted like:
        "Event Type: Low Light\nDescription: xxx\nEvent Type: Over Exposure\nDescription: yyy"
    
    Returns:
        List[str] Processed text batch
    """
    def count_tokens(text):
        return len(text.split())
    
    def truncate_single_text(text):
        events = []
        descriptions = []
        lines = text.strip().split('\n')
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                event = lines[i].strip()
                desc = lines[i + 1].strip()
                if event.startswith('Event Type:'):
                    events.append(event)
                    descriptions.append(desc)
        result = []
        for i in range(len(events)):
            event = events[i]
            event_type = event.replace('Event Type:', '').strip()
            result.extend([
                event,
                f"Description: This image has {event_type} condition."
            ])
        
        new_text = '\n'.join(result)
        while count_tokens(new_text) > 70:  
            if len(result) >= 2:
                result.pop()  
                result.pop() 
                new_text = '\n'.join(result)
            else:
                event_type = events[0].replace('Event Type:', '').strip()
                return f"Event Type: {event_type}\nDescription: Image degraded."
        
        return new_text
    
    return [truncate_single_text(text) for text in text_batch]