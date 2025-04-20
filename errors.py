from flask import Blueprint, render_template

errors = Blueprint('errors', __name__)

@errors.app_errorhandler(404)
def error_404(error):
    return render_template('errors/404.html'), 404

@errors.app_errorhandler(403)
def error_403(error):
    return render_template('errors/403.html'), 403

@errors.app_errorhandler(500)
def error_500(error):
    return render_template('errors/500.html'), 500

class SensorReadError(Exception):
    """Raised when there's an error reading from the glucose sensor"""
    pass

class PDFGenerationError(Exception):
    """Raised when there's an error generating the PDF report"""
    pass

def init_app(app):
    app.register_blueprint(errors)
    
    @app.errorhandler(SensorReadError)
    def handle_sensor_error(error):
        app.logger.error(f"Sensor error: {str(error)}")
        return {'success': False, 'error': str(error)}, 500
    
    @app.errorhandler(PDFGenerationError)
    def handle_pdf_error(error):
        app.logger.error(f"PDF generation error: {str(error)}")
        return {'success': False, 'error': str(error)}, 500 