import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  Heading,
  Text,
  Select,
  Button,
  useToast,
  Container,
  Input,
  Alert,
  AlertIcon,
} from '@chakra-ui/react';

function App() {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [models, setModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const toast = useToast();

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/api/supported-models');
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      setError(err.message);
      toast({
        title: 'Error',
        description: err.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      toast({
        title: 'File selected',
        description: `Selected file: ${file.name}`,
        status: 'info',
        duration: 3000,
      });
    }
  };

  const startTraining = async () => {
    if (!selectedModel || !selectedFile) {
      toast({
        title: 'Validation Error',
        description: 'Please select both a model and a dataset file',
        status: 'error',
        duration: 5000,
      });
      return;
    }

    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Upload dataset
      const uploadResponse = await fetch('http://localhost:8000/api/upload-dataset', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload dataset');
      }

      const uploadResult = await uploadResponse.json();

      // Start training
      const trainingResponse = await fetch('http://localhost:8000/api/start-training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: selectedModel,
          dataset_path: uploadResult.file_path,
        }),
      });

      if (!trainingResponse.ok) {
        throw new Error('Failed to start training');
      }

      toast({
        title: 'Success',
        description: 'Training started successfully',
        status: 'success',
        duration: 5000,
      });

    } catch (err) {
      toast({
        title: 'Error',
        description: err.message,
        status: 'error',
        duration: 5000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={8} align="stretch">
        <Heading textAlign="center">Fine-Tuning Labs</Heading>

        {error && (
          <Alert status="error">
            <AlertIcon />
            {error}
          </Alert>
        )}

        <Box>
          <Text mb={2} fontWeight="bold">Select Model</Text>
          <Select
            placeholder="Choose a model"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            isDisabled={isLoading}
          >
            {models.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </Select>
        </Box>

        <Box>
          <Text mb={2} fontWeight="bold">Upload Dataset</Text>
          <Input
            type="file"
            accept=".json,.jsonl,.csv,.xlsx,.txt"
            onChange={handleFileUpload}
            disabled={isLoading}
          />
        </Box>

        <Button
          colorScheme="blue"
          onClick={startTraining}
          isLoading={isLoading}
          loadingText="Processing..."
          disabled={!selectedModel || !selectedFile}
        >
          Start Training
        </Button>
      </VStack>
    </Container>
  );
}

export default App;