document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const refreshBtn = document.getElementById('refreshBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    const errorMessage = document.getElementById('errorMessage');
    const emptyState = document.getElementById('emptyState');
    const imagesList = document.getElementById('imagesList');
    const imagesContainer = document.getElementById('imagesContainer');
    const imageCardTemplate = document.getElementById('imageCardTemplate');

    // Current state for animation tracking
    let currentImageData = [];
    let unanalyzedImages = [];

    // Initialize the application
    init();

    // Event Listeners
    refreshBtn.addEventListener('click', fetchImages);
    analyzeBtn.addEventListener('click', analyzeAndSortImages);

    // Initialize the app - automatically load images
    async function init() {
        await fetchImages();
        // Do NOT automatically analyze - wait for manual button press
    }

    // Fetch images without analysis
    async function fetchImages() {
        // Show loading state
        showLoadingState();

        try {
            // Call API to get images without analysis
            const response = await fetch('/api/images');
            if (!response.ok) {
                throw new Error(`Error fetching images: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.images || data.images.length === 0) {
                showEmptyState();
                return;
            }
            
            // Store unanalyzed images for later use
            unanalyzedImages = data.images;
            
            // Show images without priority
            renderImageList(data.images);
            currentImageData = [...data.images];
            
        } catch (error) {
            console.error('Error:', error);
            showErrorState(error.message);
        }
    }
    
    // Browser popup notification
    function showToast(message, type = 'info') {
        // Just use the native browser alert
        alert(message);
    }
    
    // Check API health status
    async function checkApiHealth() {
        try {
            // Call health check API
            const response = await fetch('/api/health-check');
            if (!response.ok) {
                throw new Error(`Error checking API health: ${response.statusText}`);
            }
            
            const healthData = await response.json();
            
            // Show appropriate toast based on health status
            if (!healthData.all_healthy) {
                let message = 'API STATUS WARNING: Some API endpoints are not responding. ';
                if (!healthData.brain_api.online) {
                    message += 'Brain API is down. ';
                }
                if (!healthData.chest_api.online) {
                    message += 'Chest API is down. ';
                }
                showToast(message, 'error');
                return false;
            }
            return true;
            
        } catch (error) {
            console.error('Error:', error);
            showToast('Error checking API health: ' + error.message, 'error');
            return false;
        }
    }
    
    // Analyze and sort images with animation
    async function analyzeAndSortImages() {
        if (unanalyzedImages.length === 0) {
            console.warn('No images to analyze');
            showToast('No images available to analyze', 'warning');
            return;
        }
        
        // First check API health
        await checkApiHealth();
        // Continue regardless of health check result (alerts already shown by checkApiHealth)
        
        try {
            // Show loading indicator for analysis with more prominent styling
            const analyzeBtn = document.getElementById('analyzeBtn');
            const originalBtnText = analyzeBtn.innerHTML;
            const originalBtnClass = analyzeBtn.className;
            
            // Change button style to indicate processing
            analyzeBtn.className = 'action-button bg-yellow-500 text-white px-6 py-2.5 rounded-md font-medium text-sm shadow-lg pulse-animation';
            analyzeBtn.innerHTML = `
                <div class="flex items-center">
                    <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    ANALYZING IMAGES...
                </div>
            `;
            analyzeBtn.disabled = true;
            
            // Add a small status message above the button
            const statusMsg = document.createElement('div');
            statusMsg.id = 'analyzeStatus';
            statusMsg.className = 'text-sm text-yellow-600 font-medium mb-2';
            statusMsg.textContent = 'Analysis in progress...';
            analyzeBtn.parentNode.insertBefore(statusMsg, analyzeBtn);
            
            // Call API to analyze images
            const response = await fetch('/api/analyze');
            if (!response.ok) {
                throw new Error(`Error analyzing images: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Reset button and remove status message
            analyzeBtn.innerHTML = originalBtnText;
            analyzeBtn.className = originalBtnClass;
            analyzeBtn.disabled = false;
            
            // Remove the status message if it exists
            const statusElement = document.getElementById('analyzeStatus');
            if (statusElement) {
                statusElement.parentNode.removeChild(statusElement);
            }
            
            if (!data.results || data.results.length === 0) {
                showToast('No results returned from analysis', 'warning');
                return;
            }
            
            // Check API status and show toast if there are issues
            if (data.api_status && !data.api_status.online) {
                showToast(data.api_status.message || 'API endpoints are not responding', 'error');
            }
            
            // Check if fallbacks were used but don't show alert (already shown during health check)
            
            // Update UI with new data (with animation)
            updateImageList(data.results);
            
        } catch (error) {
            console.error('Error:', error);
            showErrorState(error.message);
        }
    }

    // Show loading state
    function showLoadingState() {
        loadingState.classList.remove('hidden');
        errorState.classList.add('hidden');
        emptyState.classList.add('hidden');
        imagesList.classList.add('hidden');
    }

    // Show error state
    function showErrorState(message) {
        loadingState.classList.add('hidden');
        errorState.classList.remove('hidden');
        emptyState.classList.add('hidden');
        imagesList.classList.add('hidden');
        errorMessage.textContent = message || 'An error occurred while loading images.';
    }

    // Show empty state
    function showEmptyState() {
        loadingState.classList.add('hidden');
        errorState.classList.add('hidden');
        emptyState.classList.remove('hidden');
        imagesList.classList.add('hidden');
    }

    // Show images list
    function showImagesList() {
        loadingState.classList.add('hidden');
        errorState.classList.add('hidden');
        emptyState.classList.add('hidden');
        imagesList.classList.remove('hidden');
    }

    // Update image list with new data and F1-style animations
    function updateImageList(imageDataToUpdate) {
        // Sort the data by priority before rendering or animating
        // Priority order: Urgent > High > Normal > Unknown
        const priorityOrder = {"urgent": 1, "high": 2, "normal": 3, "unknown": 4};
        
        // Sort by priority
        imageDataToUpdate.sort((a, b) => {
            const priorityA = a.priority ? a.priority.toLowerCase() : 'unknown';
            const priorityB = b.priority ? b.priority.toLowerCase() : 'unknown';
            
            return (priorityOrder[priorityA] || 999) - (priorityOrder[priorityB] || 999);
        });
        
        // First time loading, just render the list
        if (currentImageData.length === 0) {
            renderImageList(imageDataToUpdate);
            currentImageData = [...imageDataToUpdate];
            return;
        }

        // Compare new data with current data
        // Map images by name for easy lookup
        const currentImageMap = new Map();
        currentImageData.forEach(img => currentImageMap.set(img.name, img));

        // Create a map for new positions
        const newPositionMap = new Map();
        imageDataToUpdate.forEach((img, index) => {
            newPositionMap.set(img.name, index);
        });

        // Determine if we need animations
        const needsAnimation = currentImageData.some((img, index) => {
            const newIndex = newPositionMap.has(img.name) ? newPositionMap.get(img.name) : -1;
            return newIndex !== -1 && newIndex !== index;
        });

        if (needsAnimation) {
            animatePositionChanges(imageDataToUpdate);
        } else {
            // Just update the content if no position changes
            renderImageList(imageDataToUpdate);
        }

        // Update current data
        currentImageData = [...imageDataToUpdate];
    }

    // Render image list
    function renderImageList(imageData) {
        // Clear container
        imagesContainer.innerHTML = '';
        
        // Create new cards
        imageData.forEach(img => {
            const card = createImageCard(img);
            imagesContainer.appendChild(card);
        });
        
        // Show images list
        showImagesList();
    }

    // Create an image card from template
    function createImageCard(imageData) {
        const template = imageCardTemplate.content.cloneNode(true);
        const card = template.querySelector('.image-card');
        
        // Set card data
        const thumbnail = card.querySelector('.image-thumbnail');
        thumbnail.src = imageData.url;
        thumbnail.alt = imageData.name;
                
        // Remove file extension from image name
        const nameWithoutExtension = imageData.name.replace(/\.[^/.]+$/, '');
        card.querySelector('.image-name').textContent = nameWithoutExtension;
        
        // Handle prediction and priority info
        const predictionEl = card.querySelector('.image-prediction');
        const confidenceEl = card.querySelector('.image-confidence');
        const priorityBadge = card.querySelector('.priority-badge');
        
        // For analyzed images, show real data
        if (imageData.prediction) {
            predictionEl.textContent = imageData.prediction;
            confidenceEl.textContent = imageData.confidence ? `${imageData.confidence.toFixed(1)}%` : 'N/A';
            // Use priority directly from the API response - always use one of the three valid values
            // Fix: Ensure we never show 'Low' as it's not a valid priority from the API
            let displayPriority = imageData.priority || 'Normal';
            
            // Fix the 'Low' priority issue - convert any 'Low' to 'Normal'
            if (displayPriority.toLowerCase() === 'low') {
                displayPriority = 'Normal';
            }
            
            // Set the text content to the corrected priority
            priorityBadge.textContent = displayPriority;
            
            // Set a consistent background color based on priority
            const priority = displayPriority.toLowerCase();
            
            // Apply standard badge styling based on priority value
            if (priority === 'urgent') {
                priorityBadge.classList.add('bg-red-600');
            } else if (priority === 'high') {
                priorityBadge.classList.add('bg-yellow-500');
            } else if (priority === 'normal') {
                priorityBadge.classList.add('bg-green-500');
            } else {
                priorityBadge.classList.add('bg-gray-500');
            }
            
            // Add the border styling classes
            priorityBadge.classList.add(`priority-badge-${priority}`);
            card.classList.add(`priority-${priority}`);
        } else {
            // For unanalyzed images, infer prediction from folder name if possible
            let inferredPrediction = 'Not analyzed';
            
            // Try to infer from category or filename
            if (imageData.category) {
                // Check if this is a brain or chest image
                if (imageData.category.includes('Chest')) {
                    inferredPrediction = 'Chest Scan (unconfirmed)';
                } else if (imageData.category.includes('Glioma') || 
                           imageData.category.includes('Meningioma') || 
                           imageData.category.includes('No Tumor') || 
                           imageData.category.includes('Normal') || 
                           imageData.category.includes('Pituitary') || 
                           imageData.category.includes('Brain') || 
                           imageData.name.startsWith('Tr-') ||
                           imageData.path.includes('Brain')) {
                    inferredPrediction = 'Brain (unconfirmed)';
                }
            }
            
            predictionEl.textContent = inferredPrediction;
            confidenceEl.textContent = 'Pending analysis';
            priorityBadge.textContent = 'Click Analyze';
            priorityBadge.classList.add('bg-blue-400');
            card.classList.add('border-gray-300');
        }
        
        // Add data attribute for animations
        card.dataset.imageName = imageData.name;
        
        return card;
    }

    // Animate position changes in F1 leaderboard style
    function animatePositionChanges(imageDataToAnimate) {
        // Get current positions
        const currentCards = Array.from(imagesContainer.querySelectorAll('.image-card'));
        const currentPositions = new Map();
        
        currentCards.forEach((card, index) => {
            const imageName = card.dataset.imageName;
            currentPositions.set(imageName, index);
            
            // Store current position for animation
            card.style.position = 'relative';
            card.style.zIndex = '10';
            card.dataset.originalTop = card.offsetTop + 'px';
        });
        
        // Create new cards but keep them hidden
        imagesContainer.innerHTML = '';
        imageDataToAnimate.forEach(img => {
            const card = createImageCard(img);
            card.style.opacity = '0';
            imagesContainer.appendChild(card);
        });
        
        // Show images list
        showImagesList();
        
        // Position old cards absolutely over their original positions
        currentCards.forEach(card => {
            const imageName = card.dataset.imageName;
            
            // Skip if image is not in new data
            if (!imageDataToAnimate.some(img => img.name === imageName)) {
                return;
            }
            
            // Position the card absolutely
            card.style.position = 'absolute';
            card.style.top = card.dataset.originalTop;
            card.style.width = card.offsetWidth + 'px';
            card.style.left = '0';
            card.style.transition = 'all 0.8s ease-in-out';
            
            // Append to container
            imagesContainer.appendChild(card);
        });
        
        // Force browser reflow to ensure proper initial position
        void imagesContainer.offsetHeight;
        
        // Get new positions
        const newCards = Array.from(imagesContainer.querySelectorAll('.image-card[style="opacity: 0;"]'));
        const newCardPositions = new Map();
        
        newCards.forEach((card, index) => {
            const imageName = card.dataset.imageName;
            newCardPositions.set(imageName, {
                top: card.offsetTop,
                left: card.offsetLeft,
                width: card.offsetWidth
            });
        });
        
        // Animate old cards to new positions
        currentCards.forEach(card => {
            const imageName = card.dataset.imageName;
            
            // Skip if image is not in new data
            if (!imageDataToAnimate.some(img => img.name === imageName)) {
                card.style.opacity = '0';
                card.style.transform = 'scale(0.8)';
                setTimeout(() => card.remove(), 800);
                return;
            }
            
            const newPosition = newCardPositions.get(imageName);
            if (!newPosition) return;
            
            // Highlight cards that have changed priority
            const oldImageData = currentImageData.find(img => img.name === imageName);
            const newImageDataItem = imageDataToAnimate.find(img => img.name === imageName);
            
            if (oldImageData && newImageDataItem && oldImageData.priority !== newImageDataItem.priority) {
                card.style.animation = 'highlight 1s ease-in-out';
            }
            
            // Animate to new position
            requestAnimationFrame(() => {
                card.style.top = newPosition.top + 'px';
                card.style.width = newPosition.width + 'px';
            });
        });
        
        // After animation completes, replace with new cards
        setTimeout(() => {
            renderImageList(imageDataToAnimate);
        }, 1000);
    }

    // Manual refresh only - removed auto-refresh interval
    // The auto-refresh was causing constant image loading
    // Users can use the refresh button when needed
});
