
class Visual:
    def __init__(self, data_source):
        self.data_source = data_source

    def generate_visualization(self, data_id, viz_type):
        data = self.data_source.get_data(data_id)
        if viz_type == 'bar_chart':
            return self._create_bar_chart(data)
        elif viz_type == 'line_chart':
            return self._create_line_chart(data)
        else:
            raise ValueError("Unsupported visualization type")

    def _create_bar_chart(self, data):
        # Placeholder for bar chart generation logic
        return f"Bar chart created with data: {data}"

    def _create_line_chart(self, data):
        # Placeholder for line chart generation logic
        return f"Line chart created with data: {data}"
