document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('strokeForm');
    const predictionResult = document.getElementById('predictionResult');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        submitButton.disabled = true;

        // Gather form data
        const formData = new FormData(form);

        // Send prediction request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            // Reset button state
            submitButton.innerHTML = originalButtonText;
            submitButton.disabled = false;

            // Parse the prediction from the response
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const prediction = doc.querySelector('[data-prediction]')?.dataset.prediction;
            const error = doc.querySelector('[data-error]')?.dataset.error;

            if (error) {
                predictionResult.innerHTML = `
                    <div class="alert alert-danger">
                        <h4 class="alert-heading">Error</h4>
                        <p>${error}</p>
                    </div>
                `;
            } else if (prediction === '0') {
                predictionResult.innerHTML = `
                    <div class="alert alert-success">
                        <h4 class="alert-heading">Low Stroke Risk</h4>
                        <p>Based on the provided information, you have a low risk of stroke.</p>
                        <hr>
                        <p class="mb-0">Continue maintaining a healthy lifestyle!</p>
                    </div>
                `;
            } else {
                predictionResult.innerHTML = `
                    <div class="alert alert-danger">
                        <h4 class="alert-heading">High Stroke Risk</h4>
                        <p>Based on the provided information, you have a high risk of stroke.</p>
                        <hr>
                        <p class="mb-0">Please consult with a healthcare professional for further evaluation.</p>
                    </div>
                `;
            }

            // Scroll to the result
            predictionResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
        })
        .catch(error => {
            // Reset button state
            submitButton.innerHTML = originalButtonText;
            submitButton.disabled = false;

            // Show error
            predictionResult.innerHTML = `
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error</h4>
                    <p>An error occurred while processing your request. Please try again.</p>
                </div>
            `;
            console.error('Error:', error);
        });
    });

    // Form validation
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.checkValidity()) {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            } else {
                this.classList.remove('is-valid');
                this.classList.add('is-invalid');
            }
        });
    });
}); 